from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
	source_name: str
	source_image_url: str
	source_page_url: Optional[str] = None
	scientific_name: Optional[str] = None
	author: Optional[str] = None
	license: Optional[str] = None
	thumbnail_url: Optional[str] = None
	width: Optional[int] = None
	height: Optional[int] = None
	metadata: dict = field(default_factory=dict)


class BaseConnector(ABC):
	name: str = "base"
	rate_limit_delay: float = 0.5

	def __init__(self):
		self._semaphore = asyncio.Semaphore(5)

	@abstractmethod
	async def search(self, query: str, limit: int = 50) -> list[ImageResult]:
		pass

	async def _safe_request(self, coro):
		async with self._semaphore:
			try:
				result = await coro
				await asyncio.sleep(self.rate_limit_delay)
				return result
			except Exception as e:
				logger.warning(f"[{self.name}] request failed: {e}")
				return None
