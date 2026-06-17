import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.connectors.base import ImageResult
from app.connectors.wikimedia import WikimediaConnector
from app.connectors.inaturalist import INaturalistConnector
from app.connectors.gbif import GBIFConnector
from app.models.models import ImageAsset, UserImage, SearchHistory, Taxon

logger = logging.getLogger(__name__)

CONNECTORS = {
    "wikimedia": WikimediaConnector,
    "gbif": GBIFConnector,
    "inaturalist": INaturalistConnector,
}

_UI_LOAD = [selectinload(UserImage.asset).selectinload(ImageAsset.taxon), selectinload(UserImage.taxon)]


async def run_search(
    db: AsyncSession,
    user_id: int,
    query: str,
    sources: list[str],
    limit: int = 50,
) -> tuple[list, list[tuple[int, str]]]:
    """
    Returns (records, new_asset_ids) where:
    - records is a list of ImageRecordRead for newly added UserImages
    - new_asset_ids is [(asset_id, url)] for assets that need downloading
    """
    all_results: list[ImageResult] = []
    for source in sources:
        cls = CONNECTORS.get(source)
        if not cls:
            logger.warning(f"Unknown connector: {source}")
            continue
        connector = cls()
        results = await connector.search(query, limit=limit)
        all_results.extend(results)
        logger.info(f"[{source}] {len(results)} results for '{query}'")

    sh = SearchHistory(
        user_id=user_id,
        query=query,
        sources=",".join(sources),
        result_count=len(all_results),
    )
    db.add(sh)

    taxon = await get_or_create_taxon(db, query)

    new_asset_ids: list[tuple[int, str]] = []
    new_user_images: list[UserImage] = []

    for r in all_results:
        asset = await db.scalar(select(ImageAsset).where(ImageAsset.source_image_url == r.source_image_url))
        if not asset:
            asset = ImageAsset(
                taxon_id=taxon.id if taxon else None,
                source_name=r.source_name,
                source_image_url=r.source_image_url,
                source_page_url=r.source_page_url,
                author=r.author,
                license=r.license,
            )
            db.add(asset)
            await db.flush()
            new_asset_ids.append((asset.id, r.source_image_url))

        existing_ui = await db.scalar(select(UserImage).where(UserImage.user_id == user_id, UserImage.image_asset_id == asset.id))
        if not existing_ui:
            ui = UserImage(
                user_id=user_id,
                image_asset_id=asset.id,
                taxon_id=taxon.id if taxon else None,
                status="pending",
            )
            db.add(ui)
            new_user_images.append(ui)

    await db.flush()
    logger.info(f"[search] {len(new_user_images)} new images for user {user_id} / '{query}'")

    if new_user_images:
        ids = [ui.id for ui in new_user_images]
        res = await db.execute(select(UserImage).where(UserImage.id.in_(ids)).options(*_UI_LOAD))
        new_user_images = list(res.scalars().all())

    from app.schemas.schemas import image_record_read

    records = [image_record_read(ui) for ui in new_user_images]
    return records, new_asset_ids


async def get_or_create_taxon(db: AsyncSession, scientific_name: str) -> Taxon | None:
    taxon = await db.scalar(select(Taxon).where(Taxon.scientific_name == scientific_name))
    if not taxon:
        common_name, rank = await _fetch_taxon_meta(scientific_name)
        taxon = Taxon(scientific_name=scientific_name, common_name=common_name, rank=rank)
        db.add(taxon)
        await db.flush()
    return taxon


async def _fetch_taxon_meta(scientific_name: str) -> tuple[str | None, str | None]:
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
                if hit.get("name", "").lower() == scientific_name.lower():
                    return hit.get("preferred_common_name") or None, hit.get("rank") or None
    except Exception as exc:
        logger.debug(f"[taxon_meta] {scientific_name}: {exc}")
    return None, None
