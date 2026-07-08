import httpx
import logging
from app.connectors.base import BaseConnector, ImageResult

logger = logging.getLogger(__name__)

INATURALIST_API = "https://api.inaturalist.org/v1"
_PER_PAGE = 200  # iNaturalist API maximum
_MAX_PAGES = 10


class INaturalistConnector(BaseConnector):
    name = "inaturalist"
    rate_limit_delay = 0.5

    async def _resolve_taxon_id(self, client: httpx.AsyncClient, query: str) -> int | None:
        try:
            resp = await client.get(
                f"{INATURALIST_API}/taxa/autocomplete",
                params={"q": query, "per_page": 1},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0].get("id")
        except Exception as e:
            logger.debug(f"[inaturalist] taxon lookup failed for '{query}': {e}")
        return None

    async def search(self, query: str, limit: int = 50) -> list[ImageResult]:
        results: list[ImageResult] = []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                taxon_id = await self._resolve_taxon_id(client, query)
                if taxon_id:
                    logger.info(f"[inaturalist] resolved '{query}' → taxon_id={taxon_id}")

                for page in range(1, _MAX_PAGES + 1):
                    params: dict = {
                        "per_page": _PER_PAGE,
                        "page": page,
                        "photos": "true",
                        "license": "cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0",
                        "order_by": "votes",
                    }
                    if taxon_id:
                        params["taxon_id"] = taxon_id
                    else:
                        params["taxon_name"] = query

                    resp = await client.get(f"{INATURALIST_API}/observations", params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    obs_list = data.get("results", [])
                    if not obs_list:
                        break

                    for obs in obs_list:
                        photos = obs.get("photos", [])
                        if not photos:
                            continue
                        taxon = obs.get("taxon") or {}
                        sci_name = taxon.get("name") or query
                        user = obs.get("user", {})

                        for photo in photos[:2]:
                            url = photo.get("url", "").replace("square", "original")
                            thumb = photo.get("url", "")
                            if not url:
                                continue
                            results.append(
                                ImageResult(
                                    source_name=self.name,
                                    source_image_url=url,
                                    source_page_url=f"https://www.inaturalist.org/observations/{obs.get('id')}",
                                    scientific_name=sci_name,
                                    author=user.get("login"),
                                    license=photo.get("license_code", "").upper(),
                                    thumbnail_url=thumb,
                                    metadata={
                                        "observation_id": obs.get("id"),
                                        "quality_grade": obs.get("quality_grade"),
                                        "taxon_id": taxon.get("id"),
                                    },
                                )
                            )
                            if len(results) >= limit:
                                return results

                    total = data.get("total_results", 0)
                    if page * _PER_PAGE >= total:
                        break

        except Exception as e:
            logger.warning(f"[inaturalist] search failed for '{query}': {e}")
        return results
