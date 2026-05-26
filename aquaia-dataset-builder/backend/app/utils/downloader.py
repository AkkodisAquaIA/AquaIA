import logging

import httpx
import imagehash

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.models import ImageRecord
from app.utils.processor import process_image_bytes

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "ADIAB/1.0 (AquaIA Dataset Builder)"}


async def download_and_process(image_id: int, url: str) -> None:
	"""Download a remote image and process it (hash, thumbnail, dedup)."""
	try:
		async with httpx.AsyncClient(timeout=settings.request_timeout, follow_redirects=True) as client:
			resp = await client.get(url, headers=HEADERS)
			resp.raise_for_status()
			content = resp.content
		ext = _ext(url, resp.headers.get("content-type", ""))
	except Exception as exc:
		logger.warning(f"[downloader] image {image_id}: {exc}")
		return

	fields = await process_image_bytes(image_id, content, ext)
	if not fields:
		return

	async with AsyncSessionLocal() as db:
		try:
			record = await db.get(ImageRecord, image_id)
			if record:
				for k, v in fields.items():
					setattr(record, k, v)
				if fields.get("perceptual_hash"):
					await _flag_near_duplicate(db, image_id, fields["perceptual_hash"], fields.get("sha256_hash", ""))
			await db.commit()
		except Exception as exc:
			logger.warning(f"[downloader] DB update failed for image {image_id}: {exc}")
			await db.rollback()


async def _flag_near_duplicate(db, image_id: int, phash: str, sha256: str) -> None:
	from sqlalchemy import select

	if sha256:
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
