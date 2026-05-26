import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import ExportJob
from app.schemas.schemas import ExportRequest, ExportJobRead
from app.services.export_service import run_export

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("", response_model=list[ExportJobRead])
async def list_exports(
	limit: int = Query(20, ge=1, le=100),
	db: AsyncSession = Depends(get_db),
):
	result = await db.execute(select(ExportJob).order_by(ExportJob.created_at.desc()).limit(limit))
	return result.scalars().all()


@router.post("", response_model=ExportJobRead, status_code=201)
async def create_export(
	body: ExportRequest,
	background_tasks: BackgroundTasks,
	db: AsyncSession = Depends(get_db),
):
	job = ExportJob(
		export_type=body.export_type,
		parameters_json=json.dumps(body.parameters or {}),
		status="running",
	)
	db.add(job)
	await db.flush()
	background_tasks.add_task(run_export, job.id)
	return job


@router.get("/{job_id}/download")
async def download_export(job_id: int, db: AsyncSession = Depends(get_db)):
	job = await db.get(ExportJob, job_id)
	if not job:
		raise HTTPException(404, "Export not found")
	if job.status == "running":
		raise HTTPException(202, "Export is still being generated, try again shortly")
	if job.status != "done" or not job.output_path:
		raise HTTPException(400, "Export failed or has no output")
	path = Path(job.output_path)
	if not path.exists():
		raise HTTPException(404, "Export file not found on disk")
	return FileResponse(path, media_type="application/zip", filename=path.name)
