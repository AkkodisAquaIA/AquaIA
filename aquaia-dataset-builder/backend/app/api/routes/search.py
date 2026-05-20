from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import SearchQuery
from app.schemas.schemas import SearchRequest, SearchQueryRead, ImageRecordRead
from app.services.search_service import run_search
from app.utils.downloader import download_and_process

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=list[SearchQueryRead])
async def list_searches(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SearchQuery).order_by(SearchQuery.created_at.desc()).limit(limit)
    )
    return result.scalars().all()


@router.post("/run", response_model=list[ImageRecordRead])
async def run_search_endpoint(
    body: SearchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    records = await run_search(db, body.query, body.sources, body.limit)
    for record in records:
        background_tasks.add_task(download_and_process, record.id, record.source_image_url)
    return records
