import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models.models import Taxon, UserImage, UserTaxonReference
from app.schemas.schemas import TaxonCreate, TaxonRead

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])

INATURALIST_TAXA = "https://api.inaturalist.org/v1/taxa"


def _taxon_read(taxon: Taxon, ref_image_id: Optional[int] = None) -> TaxonRead:
    return TaxonRead(
        id=taxon.id,
        scientific_name=taxon.scientific_name,
        common_name=taxon.common_name,
        rank=taxon.rank,
        parent_taxon_id=taxon.parent_taxon_id,
        reference_image_id=ref_image_id,
        created_at=taxon.created_at,
    )


async def _get_user_ref(db: AsyncSession, user_id: int, taxon_id: int) -> Optional[int]:
    return await db.scalar(
        select(UserTaxonReference.user_image_id).where(
            UserTaxonReference.user_id == user_id,
            UserTaxonReference.taxon_id == taxon_id,
        )
    )


@router.get("/suggest", response_model=list[str])
async def suggest_taxons(
    q: str = Query(..., min_length=2, max_length=100),
    db: AsyncSession = Depends(get_db),
):
    local_result = await db.execute(select(Taxon.scientific_name).where(Taxon.scientific_name.ilike(f"{q}%")).order_by(Taxon.scientific_name).limit(5))
    local = [r[0] for r in local_result.all()]

    remote: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                INATURALIST_TAXA,
                params={"q": q, "rank": "order,family,genus,species,subspecies", "is_active": "true", "per_page": 10},
            )
            if resp.status_code == 200:
                for t in resp.json().get("results", []):
                    name = t.get("name", "")
                    if name and name not in local:
                        remote.append(name)
    except Exception as e:
        logger.warning(f"iNaturalist suggest failed: {e}")

    seen: set[str] = set(local)
    merged = list(local)
    for name in remote:
        if name not in seen:
            seen.add(name)
            merged.append(name)
    return merged[:10]


@router.get("", response_model=list[TaxonRead])
async def list_taxons(
    q: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    user_id: Optional[int] = Query(None, ge=1),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Taxon).order_by(Taxon.scientific_name).limit(limit)
    if q:
        stmt = stmt.where(Taxon.scientific_name.ilike(f"%{q}%"))
    result = await db.execute(stmt)
    taxons = result.scalars().all()

    if not user_id:
        return [_taxon_read(t) for t in taxons]

    # Load user's reference map in one query
    ref_rows = await db.execute(select(UserTaxonReference.taxon_id, UserTaxonReference.user_image_id).where(UserTaxonReference.user_id == user_id))
    ref_map = {row.taxon_id: row.user_image_id for row in ref_rows}
    return [_taxon_read(t, ref_map.get(t.id)) for t in taxons]


@router.post("", response_model=TaxonRead, status_code=201)
async def create_taxon(body: TaxonCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(select(Taxon).where(Taxon.scientific_name == body.scientific_name))
    if existing:
        raise HTTPException(409, "Taxon already exists")
    taxon = Taxon(**body.model_dump())
    db.add(taxon)
    await db.flush()
    return _taxon_read(taxon)


@router.get("/{taxon_id}", response_model=TaxonRead)
async def get_taxon(
    taxon_id: int,
    user_id: Optional[int] = Query(None, ge=1),
    db: AsyncSession = Depends(get_db),
):
    taxon = await db.get(Taxon, taxon_id)
    if not taxon:
        raise HTTPException(404, "Taxon not found")
    ref = await _get_user_ref(db, user_id, taxon_id) if user_id else None
    return _taxon_read(taxon, ref)


class ReferenceImageBody(BaseModel):
    user_id: int
    user_image_id: Optional[int] = None


@router.patch("/{taxon_id}/reference", response_model=TaxonRead)
async def set_reference_image(taxon_id: int, body: ReferenceImageBody, db: AsyncSession = Depends(get_db)):
    taxon = await db.get(Taxon, taxon_id)
    if not taxon:
        raise HTTPException(404, "Taxon not found")

    if body.user_image_id is not None:
        ui = await db.get(UserImage, body.user_image_id)
        if not ui or ui.user_id != body.user_id:
            raise HTTPException(404, "Image not found in this workspace")

    existing = await db.scalar(
        select(UserTaxonReference).where(
            UserTaxonReference.user_id == body.user_id,
            UserTaxonReference.taxon_id == taxon_id,
        )
    )
    if body.user_image_id is None:
        if existing:
            await db.delete(existing)
        return _taxon_read(taxon, None)

    if existing:
        existing.user_image_id = body.user_image_id
    else:
        db.add(UserTaxonReference(user_id=body.user_id, taxon_id=taxon_id, user_image_id=body.user_image_id))
    await db.flush()
    return _taxon_read(taxon, body.user_image_id)


@router.post("/enrich", response_model=list[TaxonRead])
async def enrich_taxons(db: AsyncSession = Depends(get_db)):
    """Fetch missing common_name / rank for all taxons that lack them."""
    from app.services.search_service import _fetch_taxon_meta

    result = await db.execute(select(Taxon).where((Taxon.common_name.is_(None)) | (Taxon.rank.is_(None))))
    taxons = list(result.scalars().all())
    updated = []
    for t in taxons:
        common, rank = await _fetch_taxon_meta(t.scientific_name)
        if common or rank:
            if common and not t.common_name:
                t.common_name = common
            if rank and not t.rank:
                t.rank = rank
            updated.append(t)
    await db.flush()
    logger.info(f"[enrich] updated {len(updated)} taxons")
    return [_taxon_read(t) for t in updated]
