from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.models.models import ImageRecord, Taxon, SearchQuery, ExportJob, Dataset
from app.schemas.schemas import DashboardStats, SearchQueryRead

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=DashboardStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    total = await db.scalar(select(func.count(ImageRecord.id))) or 0
    pending = await db.scalar(select(func.count(ImageRecord.id)).where(ImageRecord.status == "pending")) or 0
    validated = await db.scalar(select(func.count(ImageRecord.id)).where(ImageRecord.status == "validated")) or 0
    rejected = await db.scalar(select(func.count(ImageRecord.id)).where(ImageRecord.status == "rejected")) or 0
    duplicates = await db.scalar(select(func.count(ImageRecord.id)).where(ImageRecord.status == "duplicate")) or 0
    downloaded = await db.scalar(
        select(func.count(ImageRecord.id)).where(ImageRecord.local_path.isnot(None))
    ) or 0
    total_taxons = await db.scalar(select(func.count(Taxon.id))) or 0
    total_datasets = await db.scalar(select(func.count(Dataset.id))) or 0
    total_exports = await db.scalar(select(func.count(ExportJob.id))) or 0

    recent_result = await db.execute(
        select(SearchQuery).order_by(SearchQuery.created_at.desc()).limit(5)
    )
    recent_searches = recent_result.scalars().all()

    return DashboardStats(
        total_images=total,
        pending=pending,
        validated=validated,
        rejected=rejected,
        duplicates=duplicates,
        downloaded=downloaded,
        total_taxons=total_taxons,
        total_datasets=total_datasets,
        total_exports=total_exports,
        recent_searches=[SearchQueryRead.model_validate(s) for s in recent_searches],
    )
