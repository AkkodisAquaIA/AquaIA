import hashlib
import ipaddress
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete as sql_delete
from sqlalchemy.orm import selectinload

from app.api.deps import verify_workspace_access
from app.core.config import settings
from app.db.database import get_db
from app.models.models import ImageAsset, UserImage, User
from app.schemas.schemas import ImageRecordRead, ImageStatusUpdate, ImportUrlRequest, image_record_read
from app.services.search_service import get_or_create_taxon
from app.utils.downloader import download_and_process, _flag_near_duplicate
from app.utils.processor import process_image_bytes

router = APIRouter(prefix="/images", tags=["images"])

VALID_STATUSES = {"pending", "validated", "rejected", "duplicate", "review_later"}
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "tif", "tiff", "bmp"}

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
]


def _validate_url_safe(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Only http/https URLs are allowed")
    host = parsed.hostname
    if not host:
        raise HTTPException(400, "Invalid URL: missing hostname")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise HTTPException(400, f"Cannot resolve hostname: {host}")
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
            if any(ip in net for net in _PRIVATE_NETS):
                raise HTTPException(400, "URL resolves to a private or reserved address")
        except ValueError:
            pass


def _safe_path(stored: Optional[str], *roots: Path) -> Path:
    if not stored:
        raise HTTPException(404, "File not available")
    resolved = Path(stored).resolve()
    if not any(str(resolved).startswith(str(r.resolve())) for r in roots):
        raise HTTPException(403, "Access denied")
    return resolved


_UI_LOAD = [selectinload(UserImage.asset).selectinload(ImageAsset.taxon), selectinload(UserImage.taxon)]


async def _get_user_or_404(db: AsyncSession, user_id: int) -> User:
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, f"Workspace {user_id} not found")
    return user


async def _ref_for_taxon(db: AsyncSession, user_id: int, taxon_id: int | None) -> int | None:
    if not taxon_id:
        return None
    from app.models.models import UserTaxonReference

    ref = await db.scalar(
        select(UserTaxonReference.user_image_id).where(
            UserTaxonReference.user_id == user_id,
            UserTaxonReference.taxon_id == taxon_id,
        )
    )
    return ref


async def _build_read(db: AsyncSession, ui: UserImage) -> ImageRecordRead:
    ref = await _ref_for_taxon(db, ui.user_id, ui.taxon_id)
    return image_record_read(ui, ref)


# ── List ────────────────────────────────────────────────────────────────────────


@router.get("", response_model=dict)
async def list_images(
    user_id: int = Query(..., ge=1),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=1000),
    status: str | None = Query(None),
    source: str | None = Query(None),
    taxon_id: int | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    await _get_user_or_404(db, user_id)
    q = select(UserImage).where(UserImage.user_id == user_id).join(UserImage.asset).order_by(UserImage.created_at.desc())
    count_q = select(func.count(UserImage.id)).where(UserImage.user_id == user_id).join(UserImage.asset)

    if status:
        q = q.where(UserImage.status == status)
        count_q = count_q.where(UserImage.status == status)
    if source:
        q = q.where(ImageAsset.source_name == source)
        count_q = count_q.where(ImageAsset.source_name == source)
    if taxon_id:
        q = q.where(UserImage.taxon_id == taxon_id)
        count_q = count_q.where(UserImage.taxon_id == taxon_id)

    total = await db.scalar(count_q) or 0
    result = await db.execute(q.offset((page - 1) * size).limit(size).options(*_UI_LOAD))
    items_orm = result.scalars().all()
    items = [await _build_read(db, ui) for ui in items_orm]
    return {
        "items": items,
        "total": total,
        "page": page,
        "size": size,
        "pages": max(1, -(-total // size)),
    }


# ── Clear (bulk delete) ─────────────────────────────────────────────────────────


@router.delete("", status_code=204)
async def clear_images(
    user_id: int = Query(..., ge=1),
    status: str | None = Query(None),
    taxon_id: int | None = Query(None),
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(user_id, authorization, db)
    await _get_user_or_404(db, user_id)
    if status and status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of: {VALID_STATUSES}")
    q = sql_delete(UserImage).where(UserImage.user_id == user_id)
    if status:
        q = q.where(UserImage.status == status)
    if taxon_id:
        q = q.where(UserImage.taxon_id == taxon_id)
    await db.execute(q)


# ── Import by URL ───────────────────────────────────────────────────────────────


@router.post("/import-url", response_model=ImageRecordRead, status_code=201)
async def import_from_url(
    body: ImportUrlRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(body.user_id, authorization, db)
    _validate_url_safe(body.url)
    await _get_user_or_404(db, body.user_id)

    taxon = None
    if body.scientific_name:
        taxon = await get_or_create_taxon(db, body.scientific_name)

    # Get or create the shared ImageAsset
    asset = await db.scalar(select(ImageAsset).where(ImageAsset.source_image_url == body.url))
    if not asset:
        asset = ImageAsset(
            taxon_id=taxon.id if taxon else None,
            source_name="manual_url",
            source_image_url=body.url,
            source_page_url=body.url,
        )
        db.add(asset)
        await db.flush()
        background_tasks.add_task(download_and_process, asset.id, body.url)

    # Create UserImage if not already exists for this user
    existing_ui = await db.scalar(select(UserImage).where(UserImage.user_id == body.user_id, UserImage.image_asset_id == asset.id))
    if existing_ui:
        raise HTTPException(409, "This image is already in your workspace")

    ui = UserImage(
        user_id=body.user_id,
        image_asset_id=asset.id,
        taxon_id=taxon.id if taxon else None,
        status="pending",
    )
    db.add(ui)
    await db.flush()
    await db.refresh(ui, ["asset", "taxon"])
    await db.refresh(ui.asset, ["taxon"])
    return await _build_read(db, ui)


# ── Upload from disk ────────────────────────────────────────────────────────────


@router.post("/upload", response_model=list[ImageRecordRead], status_code=201)
async def upload_files(
    user_id: int = Form(...),
    files: list[UploadFile] = File(...),
    scientific_name: str | None = Form(None),
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(user_id, authorization, db)
    await _get_user_or_404(db, user_id)

    taxon = None
    if scientific_name:
        taxon = await get_or_create_taxon(db, scientific_name)

    results: list[UserImage] = []
    for upload in files:
        content = await upload.read()
        if not content:
            continue

        fname = upload.filename or ""
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "jpg"
        if ext not in IMAGE_EXTS:
            ext = "jpg"
        if ext == "jpeg":
            ext = "jpg"

        sha256 = hashlib.sha256(content).hexdigest()

        # Check for exact duplicate at asset level
        asset = await db.scalar(select(ImageAsset).where(ImageAsset.sha256_hash == sha256))
        if not asset:
            asset = ImageAsset(
                taxon_id=taxon.id if taxon else None,
                source_name="local_upload",
                source_image_url=f"upload://{fname}",
            )
            db.add(asset)
            await db.flush()
            fields = await process_image_bytes(asset.id, content, ext)
            fields["sha256_hash"] = sha256
            for k, v in fields.items():
                setattr(asset, k, v)
            if fields.get("perceptual_hash"):
                await _flag_near_duplicate(db, asset.id, fields["perceptual_hash"], sha256)

        # Create UserImage if not already exists
        existing_ui = await db.scalar(select(UserImage).where(UserImage.user_id == user_id, UserImage.image_asset_id == asset.id))
        if existing_ui:
            continue

        ui = UserImage(
            user_id=user_id,
            image_asset_id=asset.id,
            taxon_id=taxon.id if taxon else None,
            status="pending",
        )
        db.add(ui)
        results.append(ui)

    if results:
        await db.flush()
        ids = [ui.id for ui in results]
        res = await db.execute(select(UserImage).where(UserImage.id.in_(ids)).options(*_UI_LOAD))
        results = list(res.scalars().all())

    return [await _build_read(db, ui) for ui in results]


# ── Status update ───────────────────────────────────────────────────────────────


@router.patch("/{user_image_id}/status", response_model=ImageRecordRead)
async def update_status(
    user_image_id: int,
    body: ImageStatusUpdate,
    user_id: int = Query(..., ge=1),
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(user_id, authorization, db)
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of: {VALID_STATUSES}")
    ui = await db.get(UserImage, user_image_id, options=_UI_LOAD)
    if not ui or ui.user_id != user_id:
        raise HTTPException(404, "Image not found in this workspace")
    ui.status = body.status
    if body.notes is not None:
        ui.notes = body.notes
    if body.status in {"validated", "rejected", "duplicate"}:
        ui.validated_at = datetime.utcnow()
    await db.flush()
    return await _build_read(db, ui)


# ── Crop ────────────────────────────────────────────────────────────────────────


class CropRequest(BaseModel):
    x: float
    y: float
    width: float
    height: float


@router.post("/{user_image_id}/crop", response_model=ImageRecordRead)
async def crop_image(
    user_image_id: int,
    body: CropRequest,
    user_id: int = Query(..., ge=1),
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(user_id, authorization, db)
    from PIL import Image as PILImage
    import io as _io

    ui = await db.get(UserImage, user_image_id, options=_UI_LOAD)
    if not ui or ui.user_id != user_id:
        raise HTTPException(404, "Image not found in this workspace")

    asset = ui.asset
    if not asset.local_path:
        raise HTTPException(422, "Image not yet downloaded — try again in a moment")

    path = Path(asset.local_path)
    if not path.exists():
        raise HTTPException(404, "Local file not found")

    pil = PILImage.open(path).convert("RGB")
    left = max(0, int(body.x))
    top = max(0, int(body.y))
    right = min(pil.width, int(body.x + body.width))
    bottom = min(pil.height, int(body.y + body.height))
    cropped = pil.crop((left, top, right, bottom))

    buf = _io.BytesIO()
    cropped.save(buf, "JPEG", quality=92, optimize=True)
    content = buf.getvalue()

    # Create a new ImageAsset for the cropped result
    new_asset = ImageAsset(
        taxon_id=asset.taxon_id,
        source_name=asset.source_name,
        source_image_url=f"crop://{asset.source_image_url}",
    )
    db.add(new_asset)
    await db.flush()

    ext = "jpg"
    fields = await process_image_bytes(new_asset.id, content, ext)
    for k, v in fields.items():
        setattr(new_asset, k, v)

    # Point UserImage to the new cropped asset
    ui.image_asset_id = new_asset.id
    await db.flush()

    # Reload
    res = await db.execute(select(UserImage).where(UserImage.id == user_image_id).options(*_UI_LOAD))
    ui = res.scalar_one()
    return await _build_read(db, ui)


# ── File serving ─────────────────────────────────────────────────────────────────


@router.get("/{user_image_id}/file")
async def serve_image_file(
    user_image_id: int,
    user_id: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    ui = await db.get(UserImage, user_image_id, options=[selectinload(UserImage.asset)])
    if not ui or ui.user_id != user_id:
        raise HTTPException(404, "Image not found in this workspace")
    path = _safe_path(ui.asset.local_path, settings.storage_raw, settings.storage_validated, settings.storage_rejected, settings.storage_duplicates)
    if not path.exists():
        raise HTTPException(404, "File not found on disk")
    return FileResponse(path)


@router.get("/{user_image_id}/thumbnail")
async def serve_thumbnail(
    user_image_id: int,
    user_id: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    ui = await db.get(UserImage, user_image_id, options=[selectinload(UserImage.asset)])
    if not ui or ui.user_id != user_id:
        raise HTTPException(404, "Image not found in this workspace")
    path = _safe_path(ui.asset.thumbnail_path, settings.storage_thumbnails)
    if not path.exists():
        raise HTTPException(404, "Thumbnail not found on disk")
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/{user_image_id}", status_code=204)
async def delete_image(
    user_image_id: int,
    user_id: int = Query(..., ge=1),
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(user_id, authorization, db)
    ui = await db.get(UserImage, user_image_id)
    if not ui or ui.user_id != user_id:
        raise HTTPException(404, "Image not found in this workspace")
    await db.delete(ui)


@router.get("/{user_image_id}", response_model=ImageRecordRead)
async def get_image(
    user_image_id: int,
    user_id: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    ui = await db.get(UserImage, user_image_id, options=_UI_LOAD)
    if not ui or ui.user_id != user_id:
        raise HTTPException(404, "Image not found in this workspace")
    return await _build_read(db, ui)
