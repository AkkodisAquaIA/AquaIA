from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.models.models import Dataset, DatasetImage, UserImage, User
from app.schemas.schemas import DatasetCreate, DatasetRead, ImageRecordRead, image_record_read
from app.api.routes.images import _ref_for_taxon, _UI_LOAD

router = APIRouter(prefix="/datasets", tags=["datasets"])


async def _get_dataset_or_404(db: AsyncSession, dataset_id: int, user_id: int) -> Dataset:
    ds = await db.get(Dataset, dataset_id)
    if not ds or ds.user_id != user_id:
        raise HTTPException(404, "Dataset not found")
    return ds


async def _dataset_count(db: AsyncSession, dataset_id: int) -> int:
    return await db.scalar(select(func.count(DatasetImage.id)).where(DatasetImage.dataset_id == dataset_id)) or 0


@router.get("", response_model=list[DatasetRead])
async def list_datasets(user_id: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, f"Workspace {user_id} not found")
    result = await db.execute(select(Dataset).where(Dataset.user_id == user_id).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()
    out = []
    for ds in datasets:
        count = await _dataset_count(db, ds.id)
        out.append(DatasetRead(id=ds.id, user_id=ds.user_id, name=ds.name, description=ds.description, created_at=ds.created_at, image_count=count))
    return out


@router.post("", response_model=DatasetRead, status_code=201)
async def create_dataset(body: DatasetCreate, user_id: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, f"Workspace {user_id} not found")
    existing = await db.scalar(select(Dataset).where(Dataset.user_id == user_id, Dataset.name == body.name))
    if existing:
        raise HTTPException(409, f"A dataset named '{body.name}' already exists in this workspace")
    ds = Dataset(user_id=user_id, name=body.name, description=body.description)
    db.add(ds)
    await db.flush()
    return DatasetRead(id=ds.id, user_id=ds.user_id, name=ds.name, description=ds.description, created_at=ds.created_at, image_count=0)


@router.get("/assigned", response_model=dict)
async def get_assigned_images(user_id: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    """Return {user_image_id: {dataset_id, dataset_name}} for all assigned images of this user."""
    rows = await db.execute(select(DatasetImage.user_image_id, Dataset.id, Dataset.name).join(Dataset, Dataset.id == DatasetImage.dataset_id).where(Dataset.user_id == user_id))
    return {row.user_image_id: {"dataset_id": row.id, "dataset_name": row.name} for row in rows}


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: int, user_id: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    ds = await _get_dataset_or_404(db, dataset_id, user_id)
    count = await _dataset_count(db, ds.id)
    return DatasetRead(id=ds.id, user_id=ds.user_id, name=ds.name, description=ds.description, created_at=ds.created_at, image_count=count)


@router.get("/{dataset_id}/images", response_model=list[ImageRecordRead])
async def get_dataset_images(dataset_id: int, user_id: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    ds = await _get_dataset_or_404(db, dataset_id, user_id)
    result = await db.execute(select(UserImage).join(DatasetImage, DatasetImage.user_image_id == UserImage.id).where(DatasetImage.dataset_id == dataset_id).options(*_UI_LOAD))
    uis = result.scalars().all()
    return [await _build_read_for_user(db, ui) for ui in uis]


async def _build_read_for_user(db: AsyncSession, ui: UserImage) -> ImageRecordRead:
    ref = await _ref_for_taxon(db, ui.user_id, ui.taxon_id)
    return image_record_read(ui, ref)


@router.post("/{dataset_id}/images/{user_image_id}", status_code=204)
async def add_image_to_dataset(
    dataset_id: int,
    user_image_id: int,
    user_id: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    ds = await _get_dataset_or_404(db, dataset_id, user_id)
    ui = await db.get(UserImage, user_image_id)
    if not ui or ui.user_id != user_id:
        raise HTTPException(404, "Image not found in this workspace")

    # Check if already assigned to a different dataset (within this user's datasets)
    existing = await db.scalar(
        select(Dataset.name)
        .join(DatasetImage, Dataset.id == DatasetImage.dataset_id)
        .where(DatasetImage.user_image_id == user_image_id)
        .where(DatasetImage.dataset_id != dataset_id)
        .where(Dataset.user_id == user_id)
        .limit(1)
    )
    if existing:
        raise HTTPException(409, f"Image already assigned to dataset '{existing}'")

    already_here = await db.scalar(select(DatasetImage).where(DatasetImage.dataset_id == dataset_id, DatasetImage.user_image_id == user_image_id))
    if not already_here:
        db.add(DatasetImage(dataset_id=dataset_id, user_image_id=user_image_id))
        await db.flush()


@router.delete("/{dataset_id}/images/{user_image_id}", status_code=204)
async def remove_image_from_dataset(
    dataset_id: int,
    user_image_id: int,
    user_id: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    ds = await _get_dataset_or_404(db, dataset_id, user_id)
    row = await db.scalar(select(DatasetImage).where(DatasetImage.dataset_id == dataset_id, DatasetImage.user_image_id == user_image_id))
    if row:
        await db.delete(row)
        await db.flush()


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, user_id: int = Query(..., ge=1), db: AsyncSession = Depends(get_db)):
    ds = await _get_dataset_or_404(db, dataset_id, user_id)
    await db.delete(ds)
    await db.flush()
