import hashlib
import io
import logging

import httpx
import imagehash
from PIL import Image

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.models import ImageRecord

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "ADIAB/1.0 (AquaIA Dataset Builder)"}


async def download_and_process(image_id: int, url: str) -> None:
    fields: dict = {}
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=HEADERS)
            resp.raise_for_status()
            content = resp.content

        sha256 = hashlib.sha256(content).hexdigest()
        fields["sha256_hash"] = sha256
        fields["file_size"] = len(content)

        ext = _ext(url, resp.headers.get("content-type", ""))
        settings.storage_raw.mkdir(parents=True, exist_ok=True)
        raw_path = settings.storage_raw / f"{image_id}.{ext}"
        raw_path.write_bytes(content)
        fields["local_path"] = str(raw_path)

        img = Image.open(io.BytesIO(content)).convert("RGB")
        fields["width"] = img.width
        fields["height"] = img.height
        fields["perceptual_hash"] = str(imagehash.dhash(img))

        settings.storage_thumbnails.mkdir(parents=True, exist_ok=True)
        thumb = img.copy()
        thumb.thumbnail(settings.thumbnail_size, Image.LANCZOS)
        thumb_path = settings.storage_thumbnails / f"{image_id}.jpg"
        thumb.save(thumb_path, "JPEG", quality=80, optimize=True)
        fields["thumbnail_path"] = str(thumb_path)

    except Exception as exc:
        logger.warning(f"[downloader] image {image_id}: {exc}")
        return

    async with AsyncSessionLocal() as db:
        try:
            record = await db.get(ImageRecord, image_id)
            if record:
                for k, v in fields.items():
                    setattr(record, k, v)
                if fields.get("perceptual_hash"):
                    await _flag_near_duplicate(db, image_id, fields["perceptual_hash"], sha256)
            await db.commit()
        except Exception as exc:
            logger.warning(f"[downloader] DB update failed for image {image_id}: {exc}")
            await db.rollback()


async def _flag_near_duplicate(db, image_id: int, phash: str, sha256: str) -> None:
    from sqlalchemy import select
    # Exact duplicate by SHA256
    existing = await db.scalar(
        select(ImageRecord).where(
            ImageRecord.sha256_hash == sha256,
            ImageRecord.id != image_id,
        )
    )
    if existing:
        record = await db.get(ImageRecord, image_id)
        if record and record.status == "pending":
            record.status = "duplicate"
        return

    # Near-duplicate by perceptual hash (hamming distance ≤ 8)
    result = await db.execute(
        select(ImageRecord).where(
            ImageRecord.perceptual_hash.isnot(None),
            ImageRecord.id != image_id,
            ImageRecord.status != "duplicate",
        )
    )
    current_hash = imagehash.hex_to_hash(phash)
    for other in result.scalars().all():
        try:
            if abs(current_hash - imagehash.hex_to_hash(other.perceptual_hash)) <= 8:
                record = await db.get(ImageRecord, image_id)
                if record and record.status == "pending":
                    record.status = "duplicate"
                break
        except Exception:
            continue


def _ext(url: str, content_type: str) -> str:
    from_url = url.split("?")[0].rsplit(".", 1)[-1].lower()[:4]
    if from_url in ("jpg", "jpeg"):
        return "jpg"
    if from_url in ("png", "webp", "gif"):
        return from_url
    if "png" in content_type:
        return "png"
    if "webp" in content_type:
        return "webp"
    return "jpg"
