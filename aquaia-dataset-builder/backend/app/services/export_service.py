import csv
import io
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.models import DatasetImage, ExportJob, Taxon, UserImage

logger = logging.getLogger(__name__)


@dataclass
class _ExportRow:
    """Flat view of UserImage + ImageAsset used by export builders."""

    id: int  # UserImage.id
    asset_id: int
    local_path: Optional[str]
    source_name: str
    source_image_url: str
    author: Optional[str]
    license: Optional[str]
    width: Optional[int]
    height: Optional[int]
    sha256_hash: Optional[str]
    taxon: Optional[Taxon]


def _to_row(ui: UserImage) -> _ExportRow:
    a = ui.asset
    return _ExportRow(
        id=ui.id,
        asset_id=a.id,
        local_path=a.local_path,
        source_name=a.source_name,
        source_image_url=a.source_image_url,
        author=a.author,
        license=a.license,
        width=a.width,
        height=a.height,
        sha256_hash=a.sha256_hash,
        taxon=ui.taxon,
    )


async def run_export(job_id: int) -> None:
    async with AsyncSessionLocal() as db:
        try:
            job = await db.get(ExportJob, job_id)
            if not job:
                return

            if job.dataset_id:
                result = await db.execute(
                    select(UserImage)
                    .join(DatasetImage, DatasetImage.user_image_id == UserImage.id)
                    .where(DatasetImage.dataset_id == job.dataset_id)
                    .options(selectinload(UserImage.asset), selectinload(UserImage.taxon))
                )
            else:
                result = await db.execute(
                    select(UserImage).where(UserImage.user_id == job.user_id, UserImage.status == "validated").options(selectinload(UserImage.asset), selectinload(UserImage.taxon))
                )
            images = [_to_row(ui) for ui in result.scalars().all()]

            user_export_dir = settings.storage_exports / f"user_{job.user_id}"
            user_export_dir.mkdir(parents=True, exist_ok=True)
            zip_path = user_export_dir / f"export_{job_id}_{job.export_type}.zip"

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


def _taxon_label(row: _ExportRow) -> str:
    if row.taxon:
        return row.taxon.scientific_name.replace(" ", "_")
    return "unknown"


def _read_bytes(row: _ExportRow) -> bytes | None:
    if row.local_path:
        p = Path(row.local_path)
        if p.exists():
            return p.read_bytes()
    return None


def _sl_filename(folder_name: str) -> str:
    """Return the SL attribution filename for a given folder name.

    Naming convention: SL_<FOLDER_NAME_UPPERCASE>.txt
    (SL = Source / Licence)
    Format per line: <source_url> - <author> - <license>
    """
    return f"SL_{folder_name.upper()}.txt"


def _sl_txt(rows: list[_ExportRow]) -> str:
    """Build the TXT attribution file content.

    One line per image: <source_url> - <author> - <license>
    Lines are in the same order as the image files in the folder.
    """
    lines = []
    for row in rows:
        url = row.source_image_url or ""
        author = row.author or ""
        license_ = row.license or ""
        lines.append(f"{url} - {author} - {license_}")
    return "\n".join(lines) + "\n"


def _group_by_taxon(rows: list[_ExportRow]) -> dict[str, list[_ExportRow]]:
    groups: dict[str, list[_ExportRow]] = {}
    for row in rows:
        label = _taxon_label(row)
        groups.setdefault(label, []).append(row)
    return groups


def _build_classification(zip_path: Path, images: list[_ExportRow]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, rows in _group_by_taxon(images).items():
            for row in rows:
                data = _read_bytes(row)
                if data is None:
                    continue
                ext = Path(row.local_path).suffix
                zf.writestr(f"{folder}/{row.id}{ext}", data)
            zf.writestr(f"{folder}/{_sl_filename(folder)}", _sl_txt(rows))


def _build_yolo(zip_path: Path, images: list[_ExportRow]) -> None:
    valid = [(row, _read_bytes(row)) for row in images]
    valid = [(row, data) for row, data in valid if data is not None]
    taxons = sorted({_taxon_label(row) for row, _ in valid})

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, rows in _group_by_taxon([row for row, _ in valid]).items():
            for row in rows:
                data = _read_bytes(row)
                if data is None:
                    continue
                ext = Path(row.local_path).suffix
                zf.writestr(f"images/{folder}/{row.id}{ext}", data)
            zf.writestr(f"images/{folder}/{_sl_filename(folder)}", _sl_txt(rows))
        yaml = f"nc: {len(taxons)}\nnames:\n" + "".join(f"  - {t}\n" for t in taxons)
        zf.writestr("data.yaml", yaml)


def _build_coco(zip_path: Path, images: list[_ExportRow]) -> None:
    categories: list[dict] = []
    cat_index: dict[str, int] = {}
    coco_images: list[dict] = []
    annotations: list[dict] = []

    for row in images:
        if _read_bytes(row) is None:
            continue
        label = _taxon_label(row)
        if label not in cat_index:
            cat_index[label] = len(categories) + 1
            categories.append({"id": cat_index[label], "name": label, "supercategory": "invertebrate"})
        coco_images.append(
            {
                "id": row.id,
                "file_name": f"{row.id}{Path(row.local_path).suffix}",
                "width": row.width or 0,
                "height": row.height or 0,
            }
        )
        annotations.append({"id": row.id, "image_id": row.id, "category_id": cat_index[label]})

    manifest = {
        "info": {"description": "ADIAB export", "version": "1.0"},
        "categories": categories,
        "images": coco_images,
        "annotations": annotations,
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for row in images:
            data = _read_bytes(row)
            if data is None:
                continue
            zf.writestr(f"images/{row.id}{Path(row.local_path).suffix}", data)
        zf.writestr("instances_validated.json", json.dumps(manifest, indent=2))
        zf.writestr("SL_SOURCES.txt", _sl_txt(images))


def _build_csv(zip_path: Path, images: list[_ExportRow]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "taxon", "source", "url", "author", "license", "width", "height", "sha256"])
    for row in images:
        w.writerow(
            [
                row.id,
                _taxon_label(row),
                row.source_name,
                row.source_image_url,
                row.author or "",
                row.license or "",
                row.width or "",
                row.height or "",
                row.sha256_hash or "",
            ]
        )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.csv", buf.getvalue())
        zf.writestr("SL_SOURCES.txt", _sl_txt(images))
