import logging

import httpx
import imagehash

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.models import ImageAsset, UserImage
from app.utils.processor import process_image_bytes

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "ADIAB/1.0 (AquaIA Dataset Builder)"}


async def download_and_process(asset_id: int, url: str) -> None:
    """Download a remote image and update the shared ImageAsset record."""
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=HEADERS)
            resp.raise_for_status()
            content = resp.content
        ext = _ext(url, resp.headers.get("content-type", ""))
    except Exception as exc:
        logger.warning(f"[downloader] asset {asset_id}: {exc}")
        return

    fields = await process_image_bytes(asset_id, content, ext)
    if not fields:
        return

    async with AsyncSessionLocal() as db:
        try:
            asset = await db.get(ImageAsset, asset_id)
            if asset:
                for k, v in fields.items():
                    setattr(asset, k, v)
                if fields.get("perceptual_hash"):
                    await _flag_near_duplicate(db, asset_id, fields["perceptual_hash"], fields.get("sha256_hash", ""))
            await db.commit()
        except Exception as exc:
            logger.warning(f"[downloader] DB update failed for asset {asset_id}: {exc}")
            await db.rollback()


async def _flag_near_duplicate(db, asset_id: int, phash: str, sha256: str) -> None:
    from sqlalchemy import select

    duplicate_asset_id: int | None = None

    if sha256:
        existing = await db.scalar(
            select(ImageAsset.id).where(
                ImageAsset.sha256_hash == sha256,
                ImageAsset.id != asset_id,
            )
        )
        if existing:
            duplicate_asset_id = asset_id
    else:
        result = await db.execute(
            select(ImageAsset).where(
                ImageAsset.perceptual_hash.isnot(None),
                ImageAsset.id != asset_id,
            )
        )
        current_hash = imagehash.hex_to_hash(phash)
        for other in result.scalars().all():
            try:
                if abs(current_hash - imagehash.hex_to_hash(other.perceptual_hash)) <= 8:
                    duplicate_asset_id = asset_id
                    break
            except Exception:
                continue

    if duplicate_asset_id is not None:
        # Mark all UserImages pointing to this asset as duplicate
        result = await db.execute(
            select(UserImage).where(
                UserImage.image_asset_id == duplicate_asset_id,
                UserImage.status == "pending",
            )
        )
        for ui in result.scalars().all():
            ui.status = "duplicate"


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
