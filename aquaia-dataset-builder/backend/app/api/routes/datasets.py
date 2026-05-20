from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.db.database import get_db
from app.models.models import Dataset, ImageRecord
from app.schemas.schemas import DatasetCreate, DatasetRead

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetRead])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Dataset).order_by(Dataset.created_at.desc())
    )
    datasets = result.scalars().all()
    out = []
    for ds in datasets:
        count = await db.scalar(
            select(func.count()).select_from(
                Dataset.__table__.join(
                    ImageRecord.__table__,
                    Dataset.__table__.c.id == ImageRecord.__table__.c.id,
                    isouter=True
                )
            ).where(Dataset.__table__.c.id == ds.id)
        ) or 0
        # Use raw count from association table
        from app.models.models import dataset_images
        count = await db.scalar(
            select(func.count()).where(dataset_images.c.dataset_id == ds.id)
        ) or 0
        out.append(DatasetRead(
            id=ds.id,
            name=ds.name,
            description=ds.description,
            created_at=ds.created_at,
            image_count=count,
        ))
    return out


@router.post("", response_model=DatasetRead, status_code=201)
async def create_dataset(body: DatasetCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(select(Dataset).where(Dataset.name == body.name))
    if existing:
        raise HTTPException(409, f"A dataset named '{body.name}' already exists")
    ds = Dataset(name=body.name, description=body.description)
    db.add(ds)
    await db.flush()
    return DatasetRead(id=ds.id, name=ds.name, description=ds.description,
                       created_at=ds.created_at, image_count=0)


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    from app.models.models import dataset_images
    count = await db.scalar(
        select(func.count()).where(dataset_images.c.dataset_id == dataset_id)
    ) or 0
    return DatasetRead(id=ds.id, name=ds.name, description=ds.description,
                       created_at=ds.created_at, image_count=count)


@router.get("/{dataset_id}/images")
async def get_dataset_images(dataset_id: int, db: AsyncSession = Depends(get_db)):
    from app.schemas.schemas import ImageRecordRead
    ds = await db.get(Dataset, dataset_id, options=[
        selectinload(Dataset.images).selectinload(ImageRecord.taxon)
    ])
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return [ImageRecordRead.model_validate(img) for img in ds.images]


@router.post("/{dataset_id}/images/{image_id}", status_code=204)
async def add_image_to_dataset(dataset_id: int, image_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id, options=[selectinload(Dataset.images)])
    if not ds:
        raise HTTPException(404, "Dataset not found")
    img = await db.get(ImageRecord, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    if img not in ds.images:
        ds.images.append(img)
    await db.flush()


@router.delete("/{dataset_id}/images/{image_id}", status_code=204)
async def remove_image_from_dataset(dataset_id: int, image_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id, options=[selectinload(Dataset.images)])
    if not ds:
        raise HTTPException(404, "Dataset not found")
    ds.images = [i for i in ds.images if i.id != image_id]
    await db.flush()


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    await db.delete(ds)
    await db.flush()
