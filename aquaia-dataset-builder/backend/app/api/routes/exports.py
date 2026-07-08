import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import verify_workspace_access
from app.db.database import get_db
from app.models.models import ExportJob, User
from app.schemas.schemas import ExportRequest, ExportJobRead
from app.services.export_service import run_export

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("", response_model=list[ExportJobRead])
async def list_exports(
    user_id: int = Query(..., ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, f"Workspace {user_id} not found")
    result = await db.execute(select(ExportJob).where(ExportJob.user_id == user_id).order_by(ExportJob.created_at.desc()).limit(limit))
    return result.scalars().all()


@router.post("", response_model=ExportJobRead, status_code=201)
async def create_export(
    body: ExportRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    await verify_workspace_access(body.user_id, authorization, db)
    user = await db.get(User, body.user_id)
    if not user:
        raise HTTPException(404, f"Workspace {body.user_id} not found")
    job = ExportJob(
        user_id=body.user_id,
        dataset_id=body.dataset_id,
        export_type=body.export_type,
        parameters_json=json.dumps(body.parameters or {}),
        status="running",
    )
    db.add(job)
    await db.flush()
    background_tasks.add_task(run_export, job.id)
    return job


@router.get("/{job_id}/download")
async def download_export(
    job_id: int,
    user_id: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(ExportJob, job_id)
    if not job or job.user_id != user_id:
        raise HTTPException(404, "Export not found")
    if job.status == "running":
        raise HTTPException(202, "Export is still being generated, try again shortly")
    if job.status != "done" or not job.output_path:
        raise HTTPException(400, "Export failed or has no output")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(404, "Export file not found on disk")
    return FileResponse(path, media_type="application/zip", filename=path.name)
