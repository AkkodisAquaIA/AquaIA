import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import ExportJob
from app.schemas.schemas import ExportRequest, ExportJobRead

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("", response_model=list[ExportJobRead])
async def list_exports(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ExportJob).order_by(ExportJob.created_at.desc()).limit(limit)
    )
    return result.scalars().all()


@router.post("", response_model=ExportJobRead, status_code=201)
async def create_export(body: ExportRequest, db: AsyncSession = Depends(get_db)):
    job = ExportJob(
        export_type=body.export_type,
        parameters_json=json.dumps(body.parameters or {}),
        status="pending",
    )
    db.add(job)
    await db.flush()
    return job
