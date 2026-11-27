from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import verify_workspace_access
from app.db.database import get_db
from app.models.models import SearchHistory, User
from app.schemas.schemas import SearchRequest, SearchHistoryRead, ImageRecordRead
from app.services.search_service import run_search
from app.utils.downloader import download_and_process

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=list[SearchHistoryRead])
async def list_searches(
    user_id: int = Query(..., ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, f"Workspace {user_id} not found")
    result = await db.execute(select(SearchHistory).where(SearchHistory.user_id == user_id).order_by(SearchHistory.created_at.desc()).limit(limit))
    return result.scalars().all()


@router.post("/run", response_model=list[ImageRecordRead])
async def run_search_endpoint(
    body: SearchRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(body.user_id, authorization, db)
    user = await db.get(User, body.user_id)
    if not user:
        raise HTTPException(404, f"Workspace {body.user_id} not found")
    user_images, new_asset_ids = await run_search(db, body.user_id, body.query, body.sources, body.limit)
    for asset_id, url in new_asset_ids:
        background_tasks.add_task(download_and_process, asset_id, url)
    return user_images
