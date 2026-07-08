import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes import auth, datasets, exports, images, search, stats, taxonomy
from app.api.routes import users
from app.core.config import settings
from app.db.database import Base, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _pre_migrate(conn) -> bool:
    """Rename conflicting tables and add missing columns before create_all.

    Returns True if this is a legacy-schema migration, False for fresh install.
    """
    legacy = await conn.scalar(text("SELECT name FROM sqlite_master WHERE type='table' AND name='image_records'"))
    if not legacy:
        return False

    already_migrated = await conn.scalar(text("SELECT name FROM sqlite_master WHERE type='table' AND name='_legacy_dataset_images'"))
    if not already_migrated:
        # Rename the old dataset_images (has image_id) so create_all can create the new one
        old_di = await conn.scalar(text("SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_images'"))
        if old_di:
            await conn.execute(text("ALTER TABLE dataset_images RENAME TO _legacy_dataset_images"))
            logger.info("[migration] renamed dataset_images → _legacy_dataset_images")

    # Add user_id to datasets and export_jobs (silently skip if already present)
    for stmt in [
        "ALTER TABLE datasets ADD COLUMN user_id INTEGER REFERENCES users(id)",
        "ALTER TABLE export_jobs ADD COLUMN user_id INTEGER REFERENCES users(id)",
        "ALTER TABLE export_jobs ADD COLUMN dataset_id INTEGER REFERENCES datasets(id)",
    ]:
        try:
            await conn.execute(text(stmt))
        except Exception:
            pass

    return True


async def _post_migrate(conn, is_legacy: bool) -> None:
    """Copy data from old tables to new workspace-aware tables."""
    if not is_legacy:
        return

    already_migrated = await conn.scalar(text("SELECT id FROM users WHERE username='default_admin' LIMIT 1"))
    if already_migrated:
        logger.info("[migration] default_admin workspace already exists — skipping data copy")
        return

    logger.info("[migration] Migrating data to workspace model…")

    # Create default_admin workspace
    await conn.execute(text("INSERT INTO users (username, display_name, created_at, updated_at) VALUES ('default_admin', 'Admin', datetime('now'), datetime('now'))"))
    admin_id = await conn.scalar(text("SELECT id FROM users WHERE username='default_admin'"))

    # Copy image_records → image_assets
    await conn.execute(
        text(
            "INSERT OR IGNORE INTO image_assets "
            "(id, taxon_id, source_name, source_image_url, source_page_url, author, license, "
            "local_path, thumbnail_path, width, height, file_size, sha256_hash, perceptual_hash, created_at) "
            "SELECT id, taxon_id, source_name, source_image_url, source_page_url, author, license, "
            "local_path, thumbnail_path, width, height, file_size, sha256_hash, perceptual_hash, created_at "
            "FROM image_records"
        )
    )

    # Create user_images from image_records (all assigned to default_admin)
    await conn.execute(
        text(
            f"INSERT OR IGNORE INTO user_images "
            f"(user_id, image_asset_id, taxon_id, status, notes, created_at, validated_at) "
            f"SELECT {admin_id}, id, taxon_id, status, notes, created_at, validated_at "
            f"FROM image_records"
        )
    )

    # Assign all existing datasets to default_admin
    await conn.execute(text(f"UPDATE datasets SET user_id = {admin_id} WHERE user_id IS NULL"))

    # Rebuild dataset_images using new user_image IDs
    legacy_di = await conn.scalar(text("SELECT name FROM sqlite_master WHERE type='table' AND name='_legacy_dataset_images'"))
    if legacy_di:
        await conn.execute(
            text(
                f"INSERT OR IGNORE INTO dataset_images (dataset_id, user_image_id, created_at) "
                f"SELECT ld.dataset_id, ui.id, ld.created_at "
                f"FROM _legacy_dataset_images ld "
                f"JOIN user_images ui ON ui.image_asset_id = ld.image_id AND ui.user_id = {admin_id}"
            )
        )

    # Copy search_queries → search_history (old table name was search_queries)
    sq_exists = await conn.scalar(text("SELECT name FROM sqlite_master WHERE type='table' AND name='search_queries'"))
    if sq_exists:
        await conn.execute(
            text(f"INSERT OR IGNORE INTO search_history (user_id, query, sources, result_count, created_at) SELECT {admin_id}, query, source, result_count, created_at FROM search_queries")
        )

    # Migrate taxon reference images → user_taxon_references
    # Old taxons.reference_image_id → image_records.id → image_assets.id → user_images.id
    ref_col = await conn.scalar(text("SELECT name FROM pragma_table_info('taxons') WHERE name='reference_image_id'"))
    if ref_col:
        await conn.execute(
            text(
                f"INSERT OR IGNORE INTO user_taxon_references (user_id, taxon_id, user_image_id, created_at) "
                f"SELECT {admin_id}, t.id, ui.id, datetime('now') "
                f"FROM taxons t "
                f"JOIN user_images ui ON ui.image_asset_id = t.reference_image_id AND ui.user_id = {admin_id} "
                f"WHERE t.reference_image_id IS NOT NULL"
            )
        )

    # Assign all existing export_jobs to default_admin
    await conn.execute(text(f"UPDATE export_jobs SET user_id = {admin_id} WHERE user_id IS NULL"))

    logger.info(f"[migration] Workspace migration complete — admin_id={admin_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ADIAB backend")
    async with engine.begin() as conn:
        is_legacy = await _pre_migrate(conn)
        await conn.run_sync(Base.metadata.create_all)
        await _post_migrate(conn, is_legacy)

        # Ensure at least one workspace exists (fresh install case)
        count = await conn.scalar(text("SELECT COUNT(*) FROM users"))
        if not count:
            await conn.execute(text("INSERT INTO users (username, display_name, created_at, updated_at) VALUES ('default_admin', 'Admin', datetime('now'), datetime('now'))"))
            logger.info("[startup] Created default_admin workspace")

        # Security column — safe to run every startup (fails silently if already present)
        try:
            await conn.execute(text("ALTER TABLE users ADD COLUMN password_hash TEXT"))
            logger.info("[migration] Added users.password_hash column")
        except Exception:
            pass

    for path in [
        settings.storage_raw,
        settings.storage_thumbnails,
        settings.storage_exports,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    logger.info("ADIAB backend ready")
    yield
    await engine.dispose()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(taxonomy.router, prefix="/api")
app.include_router(exports.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version}
