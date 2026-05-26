import httpx
import logging
from app.connectors.base import BaseConnector, ImageResult

logger = logging.getLogger(__name__)

GBIF_API = "https://api.gbif.org/v1"

CC_LICENSES = {
	"http://creativecommons.org/licenses/by/4.0/": "CC-BY",
	"http://creativecommons.org/licenses/by-nc/4.0/": "CC-BY-NC",
	"http://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA",
	"http://creativecommons.org/licenses/by-nc-sa/4.0/": "CC-BY-NC-SA",
	"http://creativecommons.org/publicdomain/zero/1.0/": "CC0",
}


class GBIFConnector(BaseConnector):
	name = "gbif"
	rate_limit_delay = 0.3

	async def search(self, query: str, limit: int = 50) -> list[ImageResult]:
		results: list[ImageResult] = []
		try:
			async with httpx.AsyncClient(timeout=30) as client:
				resp = await client.get(
					f"{GBIF_API}/occurrence/search",
					params={
						"q": query,
						"mediaType": "StillImage",
						"limit": min(limit, 300),
						"license": ",".join(CC_LICENSES.keys()),
					},
				)
				resp.raise_for_status()
				data = resp.json()

				for occ in data.get("results", []):
					sci_name = occ.get("scientificName") or query
					occ_id = occ.get("key")
					page_url = f"https://www.gbif.org/occurrence/{occ_id}" if occ_id else None

					for media in occ.get("media", []):
						if media.get("type") != "StillImage":
							continue
						url = media.get("identifier", "")
						if not url or not url.startswith("http"):
							continue

						raw_license = media.get("license", "")
						license_name = CC_LICENSES.get(raw_license, raw_license.split("/")[-2].upper() if raw_license else None)

						results.append(
							ImageResult(
								source_name=self.name,
								source_image_url=url,
								source_page_url=media.get("references") or page_url,
								scientific_name=sci_name,
								author=media.get("creator"),
								license=license_name,
								thumbnail_url=url,
								metadata={
									"occurrence_id": occ_id,
									"country": occ.get("countryCode"),
									"year": occ.get("year"),
								},
							)
						)

						if len(results) >= limit:
							return results

		except Exception as e:
			logger.warning(f"[gbif] search failed for '{query}': {e}")
		return results
