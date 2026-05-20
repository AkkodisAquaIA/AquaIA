from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import engine, Base
from app.api.routes import search, images, taxonomy, exports, stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ADIAB backend — creating tables")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    for path in [
        settings.storage_raw, settings.storage_validated, settings.storage_rejected,
        settings.storage_duplicates, settings.storage_exports, settings.storage_thumbnails,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    logger.info("ADIAB backend ready")
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stats.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(taxonomy.router, prefix="/api")
app.include_router(exports.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version}
