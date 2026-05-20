import csv
import io
import json
import logging
import zipfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.models import ExportJob, ImageRecord

logger = logging.getLogger(__name__)


async def run_export(job_id: int) -> None:
    async with AsyncSessionLocal() as db:
        try:
            job = await db.get(ExportJob, job_id)
            if not job:
                return

            result = await db.execute(
                select(ImageRecord)
                .where(ImageRecord.status == "validated")
                .options(selectinload(ImageRecord.taxon))
            )
            images = result.scalars().all()

            settings.storage_exports.mkdir(parents=True, exist_ok=True)
            zip_path = settings.storage_exports / f"export_{job_id}_{job.export_type}.zip"

            builder = {
                "classification": _build_classification,
                "yolo": _build_yolo,
                "coco": _build_coco,
                "csv": _build_csv,
            }.get(job.export_type, _build_classification)

            builder(zip_path, images)

            job.output_path = str(zip_path)
            job.status = "done"
            await db.commit()
            logger.info(f"[export] job {job_id} ({job.export_type}) → {zip_path}")

        except Exception as exc:
            logger.error(f"[export] job {job_id} failed: {exc}")
            try:
                job = await db.get(ExportJob, job_id)
                if job:
                    job.status = "error"
                await db.commit()
            except Exception:
                pass


def _taxon_label(img: ImageRecord) -> str:
    if img.taxon:
        return img.taxon.scientific_name.replace(" ", "_")
    return "unknown"


def _read_bytes(img: ImageRecord) -> bytes | None:
    if img.local_path:
        p = Path(img.local_path)
        if p.exists():
            return p.read_bytes()
    return None


def _build_classification(zip_path: Path, images: list) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            data = _read_bytes(img)
            if data is None:
                continue
            ext = Path(img.local_path).suffix
            zf.writestr(f"{_taxon_label(img)}/{img.id}{ext}", data)


def _build_yolo(zip_path: Path, images: list) -> None:
    valid = [(img, _read_bytes(img)) for img in images]
    valid = [(img, data) for img, data in valid if data is not None]
    taxons = sorted({_taxon_label(img) for img, _ in valid})

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img, data in valid:
            ext = Path(img.local_path).suffix
            zf.writestr(f"images/{_taxon_label(img)}/{img.id}{ext}", data)
        yaml = f"nc: {len(taxons)}\nnames:\n" + "".join(f"  - {t}\n" for t in taxons)
        zf.writestr("data.yaml", yaml)


def _build_coco(zip_path: Path, images: list) -> None:
    categories: list[dict] = []
    cat_index: dict[str, int] = {}
    coco_images: list[dict] = []
    annotations: list[dict] = []

    for img in images:
        if _read_bytes(img) is None:
            continue
        label = _taxon_label(img)
        if label not in cat_index:
            cat_index[label] = len(categories) + 1
            categories.append({"id": cat_index[label], "name": label, "supercategory": "invertebrate"})
        coco_images.append({
            "id": img.id,
            "file_name": f"{img.id}{Path(img.local_path).suffix}",
            "width": img.width or 0,
            "height": img.height or 0,
        })
        annotations.append({"id": img.id, "image_id": img.id, "category_id": cat_index[label]})

    manifest = {
        "info": {"description": "ADIAB export", "version": "1.0"},
        "categories": categories,
        "images": coco_images,
        "annotations": annotations,
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            data = _read_bytes(img)
            if data is None:
                continue
            zf.writestr(f"images/{img.id}{Path(img.local_path).suffix}", data)
        zf.writestr("instances_validated.json", json.dumps(manifest, indent=2))


def _build_csv(zip_path: Path, images: list) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "taxon", "source", "url", "author", "license", "width", "height", "sha256"])
    for img in images:
        w.writerow([
            img.id, _taxon_label(img), img.source_name, img.source_image_url,
            img.author or "", img.license or "",
            img.width or "", img.height or "", img.sha256_hash or "",
        ])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.csv", buf.getvalue())
