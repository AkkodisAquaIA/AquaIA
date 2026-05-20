from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import Taxon
from app.schemas.schemas import TaxonCreate, TaxonRead

router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])


@router.get("", response_model=list[TaxonRead])
async def list_taxons(
    q: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Taxon).order_by(Taxon.scientific_name).limit(limit)
    if q:
        stmt = stmt.where(Taxon.scientific_name.ilike(f"%{q}%"))
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("", response_model=TaxonRead, status_code=201)
async def create_taxon(body: TaxonCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(
        select(Taxon).where(Taxon.scientific_name == body.scientific_name)
    )
    if existing:
        raise HTTPException(409, "Taxon already exists")
    taxon = Taxon(**body.model_dump())
    db.add(taxon)
    await db.flush()
    return taxon


@router.get("/{taxon_id}", response_model=TaxonRead)
async def get_taxon(taxon_id: int, db: AsyncSession = Depends(get_db)):
    taxon = await db.get(Taxon, taxon_id)
    if not taxon:
        raise HTTPException(404, "Taxon not found")
    return taxon
