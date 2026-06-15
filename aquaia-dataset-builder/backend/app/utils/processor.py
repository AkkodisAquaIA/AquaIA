import hashlib
import io
import logging

import imagehash
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


async def process_image_bytes(image_id: int, content: bytes, ext: str = "jpg") -> dict:
    """Save raw file, compute hashes, generate thumbnail. Returns dict of fields."""
    fields: dict = {}
    fields["sha256_hash"] = hashlib.sha256(content).hexdigest()
    fields["file_size"] = len(content)

    settings.storage_raw.mkdir(parents=True, exist_ok=True)
    raw_path = settings.storage_raw / f"{image_id}.{ext}"
    raw_path.write_bytes(content)
    fields["local_path"] = str(raw_path)

    try:
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
        logger.warning(f"[processor] image processing failed for {image_id}: {exc}")

    return fields
