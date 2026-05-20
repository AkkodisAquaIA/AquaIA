import json
import logging
from datetime import datetime
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.connectors.base import ImageResult
from app.connectors.wikimedia import WikimediaConnector
from app.connectors.inaturalist import INaturalistConnector
from app.connectors.gbif import GBIFConnector
from app.models.models import ImageRecord, SearchQuery, Taxon

logger = logging.getLogger(__name__)

CONNECTORS = {
    "wikimedia": WikimediaConnector,
    "gbif": GBIFConnector,
    "inaturalist": INaturalistConnector,
}


async def run_search(
    db: AsyncSession,
    query: str,
    sources: list[str],
    limit: int = 50,
) -> list[ImageRecord]:
    all_results: list[ImageResult] = []

    for source in sources:
        cls = CONNECTORS.get(source)
        if not cls:
            logger.warning(f"Unknown connector: {source}")
            continue
        connector = cls()
        results = await connector.search(query, limit=limit)
        all_results.extend(results)
        logger.info(f"[{source}] returned {len(results)} results for '{query}'")

    # Persist search query
    sq = SearchQuery(
        query=query,
        source=",".join(sources),
        result_count=len(all_results),
        filters_json=json.dumps({"sources": sources, "limit": limit}),
    )
    db.add(sq)

    # Upsert taxon
    taxon = await get_or_create_taxon(db, query)

    # Persist image records (skip exact URL duplicates)
    saved: list[ImageRecord] = []
    for r in all_results:
        existing = await db.scalar(
            select(ImageRecord).where(ImageRecord.source_image_url == r.source_image_url)
        )
        if existing:
            continue
        img = ImageRecord(
            taxon_id=taxon.id if taxon else None,
            source_name=r.source_name,
            source_image_url=r.source_image_url,
            source_page_url=r.source_page_url,
            author=r.author,
            license=r.license,
            status="pending",
        )
        db.add(img)
        saved.append(img)

    await db.flush()
    logger.info(f"Saved {len(saved)} new image records for '{query}'")

    # Reload with eager taxon to avoid lazy-load in async serialization
    if saved:
        ids = [r.id for r in saved]
        result = await db.execute(
            select(ImageRecord)
            .where(ImageRecord.id.in_(ids))
            .options(selectinload(ImageRecord.taxon))
        )
        saved = list(result.scalars().all())

    return saved


async def get_or_create_taxon(db: AsyncSession, scientific_name: str) -> Taxon | None:
    taxon = await db.scalar(
        select(Taxon).where(Taxon.scientific_name == scientific_name)
    )
    if not taxon:
        common_name, rank = await _fetch_taxon_meta(scientific_name)
        taxon = Taxon(scientific_name=scientific_name, common_name=common_name, rank=rank)
        db.add(taxon)
        await db.flush()
    return taxon


async def _fetch_taxon_meta(scientific_name: str) -> tuple[str | None, str | None]:
    """Fetch common name and rank from iNaturalist taxa API."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.inaturalist.org/v1/taxa/autocomplete",
                params={"q": scientific_name, "per_page": 1},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                hit = results[0]
                # Only use result if name matches closely
                if hit.get("name", "").lower() == scientific_name.lower():
                    common = hit.get("preferred_common_name") or None
                    rank = hit.get("rank") or None
                    return common, rank
    except Exception as exc:
        logger.debug(f"[taxon_meta] {scientific_name}: {exc}")
    return None, None
