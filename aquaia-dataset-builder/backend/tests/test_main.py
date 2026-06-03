import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_health():
	async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
		resp = await client.get("/health")
	assert resp.status_code == 200
	assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_stats_empty():
	async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
		resp = await client.get("/api/stats")
	assert resp.status_code == 200
	data = resp.json()
	assert "total_images" in data
	assert data["total_images"] == 0
