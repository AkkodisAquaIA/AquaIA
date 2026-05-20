from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.models.models import ImageRecord
from app.schemas.schemas import ImageRecordRead, ImageStatusUpdate

router = APIRouter(prefix="/images", tags=["images"])

VALID_STATUSES = {"pending", "validated", "rejected", "duplicate", "review_later"}


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
    result = await db.execute(q.offset((page - 1) * size).limit(size))
    items = result.scalars().all()

    return {
        "items": [ImageRecordRead.model_validate(i) for i in items],
        "total": total,
        "page": page,
        "size": size,
        "pages": max(1, -(-total // size)),
    }


@router.patch("/{image_id}/status", response_model=ImageRecordRead)
async def update_status(
    image_id: int,
    body: ImageStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"Invalid status. Must be one of: {VALID_STATUSES}")

    img = await db.get(ImageRecord, image_id)
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


@router.get("/{image_id}/file")
async def serve_image_file(image_id: int, db: AsyncSession = Depends(get_db)):
    img = await db.get(ImageRecord, image_id)
    if not img or not img.local_path:
        raise HTTPException(404, "Local file not available — download may be in progress")
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
    img = await db.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    return ImageRecordRead.model_validate(img)
