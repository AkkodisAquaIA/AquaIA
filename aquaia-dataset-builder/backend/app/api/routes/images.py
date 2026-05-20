import hashlib
from datetime import datetime
from pathlib import Path

import imagehash
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.db.database import get_db
from app.models.models import ImageRecord
from app.schemas.schemas import ImageRecordRead, ImageStatusUpdate, ImportUrlRequest
from app.services.search_service import get_or_create_taxon
from app.utils.downloader import download_and_process, _flag_near_duplicate
from app.utils.processor import process_image_bytes

router = APIRouter(prefix="/images", tags=["images"])

VALID_STATUSES = {"pending", "validated", "rejected", "duplicate", "review_later"}
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "tif", "tiff", "bmp"}


# ── List ────────────────────────────────────────────────────────────────────

@router.get("", response_model=dict)
async def list_images(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    status: str | None = Query(None),
    source: str | None = Query(None),
    taxon_id: int | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    q = select(ImageRecord).order_by(ImageRecord.created_at.desc())
    count_q = select(func.count(ImageRecord.id))
    if status:
        q = q.where(ImageRecord.status == status)
        count_q = count_q.where(ImageRecord.status == status)
    if source:
        q = q.where(ImageRecord.source_name == source)
        count_q = count_q.where(ImageRecord.source_name == source)
    if taxon_id:
        q = q.where(ImageRecord.taxon_id == taxon_id)
        count_q = count_q.where(ImageRecord.taxon_id == taxon_id)

    total = await db.scalar(count_q) or 0
    result = await db.execute(
        q.offset((page - 1) * size).limit(size).options(selectinload(ImageRecord.taxon))
    )
    items = result.scalars().all()
    return {
        "items": [ImageRecordRead.model_validate(i) for i in items],
        "total": total,
        "page": page,
        "size": size,
        "pages": max(1, -(-total // size)),
    }


# ── Import by URL ───────────────────────────────────────────────────────────

@router.post("/import-url", response_model=ImageRecordRead, status_code=201)
async def import_from_url(
    body: ImportUrlRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    existing = await db.scalar(
        select(ImageRecord).where(ImageRecord.source_image_url == body.url)
    )
    if existing:
        raise HTTPException(409, "This URL is already in the database")

    taxon = None
    if body.scientific_name:
        taxon = await get_or_create_taxon(db, body.scientific_name)

    img = ImageRecord(
        taxon_id=taxon.id if taxon else None,
        source_name="manual_url",
        source_image_url=body.url,
        source_page_url=body.url,
        status="pending",
    )
    db.add(img)
    await db.flush()
    background_tasks.add_task(download_and_process, img.id, img.source_image_url)
    # Reload with taxon to avoid lazy-load error during serialization
    await db.refresh(img, ["taxon"])
    return ImageRecordRead.model_validate(img)


# ── Upload from disk ────────────────────────────────────────────────────────

@router.post("/upload", response_model=list[ImageRecordRead], status_code=201)
async def upload_files(
    files: list[UploadFile] = File(...),
    scientific_name: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    taxon = None
    if scientific_name:
        taxon = await get_or_create_taxon(db, scientific_name)

    records: list[ImageRecord] = []
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
        dupe = await db.scalar(
            select(ImageRecord).where(ImageRecord.sha256_hash == sha256)
        )
        if dupe:
            continue

        img = ImageRecord(
            taxon_id=taxon.id if taxon else None,
            source_name="local_upload",
            source_image_url=f"upload://{fname}",
            status="pending",
        )
        db.add(img)
        await db.flush()

        fields = await process_image_bytes(img.id, content, ext)
        fields["sha256_hash"] = sha256
        for k, v in fields.items():
            setattr(img, k, v)

        if fields.get("perceptual_hash"):
            await _flag_near_duplicate(db, img.id, fields["perceptual_hash"], sha256)

        records.append(img)

    # Reload all with taxon eager-loaded
    if records:
        ids = [r.id for r in records]
        result = await db.execute(
            select(ImageRecord)
            .where(ImageRecord.id.in_(ids))
            .options(selectinload(ImageRecord.taxon))
        )
        records = list(result.scalars().all())

    return [ImageRecordRead.model_validate(r) for r in records]


# ── Status update ───────────────────────────────────────────────────────────

@router.patch("/{image_id}/status", response_model=ImageRecordRead)
async def update_status(
    image_id: int,
    body: ImageStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of: {VALID_STATUSES}")
    img = await db.get(ImageRecord, image_id, options=[selectinload(ImageRecord.taxon)])
    if not img:
        raise HTTPException(404, "Image not found")
    img.status = body.status
    if body.notes is not None:
        img.notes = body.notes
    if body.status in {"validated", "rejected", "duplicate"}:
        img.validated_at = datetime.utcnow()
        img.validated_by = body.validated_by or "user"
    await db.flush()
    return ImageRecordRead.model_validate(img)


# ── File serving ────────────────────────────────────────────────────────────

@router.post("/{image_id}/crop", response_model=ImageRecordRead)
async def crop_image(
    image_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    img = await db.get(ImageRecord, image_id, options=[selectinload(ImageRecord.taxon)])
    if not img:
        raise HTTPException(404, "Image not found")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    ext = (img.local_path or "x.jpg").rsplit(".", 1)[-1].lower()
    if ext not in IMAGE_EXTS:
        ext = "jpg"
    fields = await process_image_bytes(image_id, content, ext)
    fields["sha256_hash"] = __import__("hashlib").sha256(content).hexdigest()
    for k, v in fields.items():
        setattr(img, k, v)
    await db.flush()
    return ImageRecordRead.model_validate(img)


@router.get("/{image_id}/file")
async def serve_image_file(image_id: int, db: AsyncSession = Depends(get_db)):
    img = await db.get(ImageRecord, image_id)
    if not img or not img.local_path:
        raise HTTPException(404, "Local file not available")
    path = Path(img.local_path)
    if not path.exists():
        raise HTTPException(404, "File not found on disk")
    return FileResponse(path)


@router.get("/{image_id}/thumbnail")
async def serve_thumbnail(image_id: int, db: AsyncSession = Depends(get_db)):
    img = await db.get(ImageRecord, image_id)
    if not img or not img.thumbnail_path:
        raise HTTPException(404, "Thumbnail not available")
    path = Path(img.thumbnail_path)
    if not path.exists():
        raise HTTPException(404, "Thumbnail not found on disk")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{image_id}", response_model=ImageRecordRead)
async def get_image(image_id: int, db: AsyncSession = Depends(get_db)):
    img = await db.get(ImageRecord, image_id, options=[selectinload(ImageRecord.taxon)])
    if not img:
        raise HTTPException(404, "Image not found")
    return ImageRecordRead.model_validate(img)
