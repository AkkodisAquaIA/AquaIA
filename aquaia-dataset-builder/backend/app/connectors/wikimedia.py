import httpx
import logging
from app.connectors.base import BaseConnector, ImageResult

logger = logging.getLogger(__name__)

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


class WikimediaConnector(BaseConnector):
	name = "wikimedia"
	rate_limit_delay = 0.3

	async def search(self, query: str, limit: int = 50) -> list[ImageResult]:
		results: list[ImageResult] = []
		try:
			async with httpx.AsyncClient(timeout=30) as client:
				params = {
					"action": "query",
					"generator": "search",
					"gsrsearch": f"{query} filetype:bitmap",
					"gsrnamespace": 6,
					"gsrlimit": min(limit, 50),
					"prop": "imageinfo",
					"iiprop": "url|size|mime|extmetadata",
					"format": "json",
				}
				resp = await client.get(WIKIMEDIA_API, params=params)
				resp.raise_for_status()
				data = resp.json()

				pages = data.get("query", {}).get("pages", {})
				for page in pages.values():
					info_list = page.get("imageinfo", [])
					if not info_list:
						continue
					info = info_list[0]
					url = info.get("url", "")
					if not url or info.get("mime", "").split("/")[0] != "image":
						continue

					meta = info.get("extmetadata", {})
					artist = meta.get("Artist", {}).get("value", "")
					license_name = meta.get("LicenseShortName", {}).get("value", "")
					thumb = info.get("thumburl") or url

					results.append(
						ImageResult(
							source_name=self.name,
							source_image_url=url,
							source_page_url=info.get("descriptionurl"),
							scientific_name=query,
							author=artist[:200] if artist else None,
							license=license_name[:100] if license_name else None,
							thumbnail_url=thumb,
							width=info.get("width"),
							height=info.get("height"),
							metadata={"mime": info.get("mime")},
						)
					)
		except Exception as e:
			logger.warning(f"[wikimedia] search failed for '{query}': {e}")
		return results
