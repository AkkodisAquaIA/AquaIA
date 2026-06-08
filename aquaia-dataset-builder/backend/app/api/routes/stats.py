from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.models.models import UserImage, ImageAsset, Taxon, SearchHistory, ExportJob, Dataset, User
from app.schemas.schemas import DashboardStats, SearchHistoryRead

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=DashboardStats)
async def get_stats(user_id: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, f"Workspace {user_id} not found")

    base = select(func.count(UserImage.id)).where(UserImage.user_id == user_id)
    total = await db.scalar(base) or 0
    pending = await db.scalar(base.where(UserImage.status == "pending")) or 0
    validated = await db.scalar(base.where(UserImage.status == "validated")) or 0
    rejected = await db.scalar(base.where(UserImage.status == "rejected")) or 0
    duplicates = await db.scalar(base.where(UserImage.status == "duplicate")) or 0
    downloaded = await db.scalar(select(func.count(UserImage.id)).where(UserImage.user_id == user_id).join(UserImage.asset).where(ImageAsset.local_path.isnot(None))) or 0
    total_taxons = await db.scalar(select(func.count(Taxon.id))) or 0
    total_datasets = await db.scalar(select(func.count(Dataset.id)).where(Dataset.user_id == user_id)) or 0
    total_exports = await db.scalar(select(func.count(ExportJob.id)).where(ExportJob.user_id == user_id)) or 0

    recent_result = await db.execute(select(SearchHistory).where(SearchHistory.user_id == user_id).order_by(SearchHistory.created_at.desc()).limit(5))
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
        recent_searches=[SearchHistoryRead.model_validate(s) for s in recent_searches],
    )
