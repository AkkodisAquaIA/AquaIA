# ADIAB — AquaIA Dataset Builder

Professional AI dataset platform for aquatic macro-invertebrate identification.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## Docker commands

| Action | Command |
|--------|---------|
| Start (with rebuild) | `docker compose up --build` |
| Start (background) | `docker compose up --build -d` |
| Stop & remove containers | `docker compose down` |
| Stop & remove containers + images | `docker compose down --rmi local` |
| View logs | `docker compose logs -f` |
| View running containers | `docker compose ps` |

## Workflow

```
1. Search          → images fetched from sources, saved as "pending"
                          ↓
2. Validation Queue → manually review: validate ✅ / reject ❌ / duplicate
                          ↓
3. Dataset Explorer → only "validated" images appear here, organised in datasets
                          ↓
4. Export Center   → export validated images as YOLO / COCO / CSV / Classification
```

The 50 images returned by a search land directly in **Validation Queue → Pending tab**. Nothing is included in a dataset until you explicitly validate it.

## Usage

### 1. Search — retrieve images

- Click **Search** in the sidebar
- Type a scientific name: `Ephemeroptera`, `Baetis rhodani`, `Plecoptera`…
- Select sources: **Wikimedia Commons**, **iNaturalist**, **GBIF**
- Or add a single image by URL, or upload files from your computer
- Click **Search** → images are fetched and saved to the database with status `pending`

### 2. Validation Queue — review images

- Click **Validation Queue** — all `pending` images are waiting here
- Click an image to select it, then use buttons or keyboard shortcuts:
  - `V` — Validate ✅ (image moves to your dataset)
  - `R` — Reject ❌ (excluded from exports)
  - `D` — Mark as duplicate
  - `Space` — Deselect
- Use the filter tabs to browse Pending / Validated / Rejected / Duplicates

### 3. Dataset Explorer — organise validated images

- **Datasets tab**: create named collections and group your validated images
- **Validated tab**: browse all validated images as a grid
- **Taxons tab**: view the taxonomy built automatically from your searches

### 4. Export Center — export for AI training

- Choose a format: **Classification** (folder-per-class) · **YOLO** · **COCO JSON** · **CSV**
- Click **Create export job** → a zip is generated in the background
- Click **Download** once the job status shows `done`

## Storage

Images are stored locally under `aquaia-dataset-builder/storage/` (never committed to git):

```
storage/
├── raw/          ← downloaded images (named by DB id, e.g. 42.jpg)
├── thumbnails/   ← auto-generated 256×256 previews
├── exports/      ← generated zip files
└── adiab.db      ← SQLite database
```

## Architecture

```
aquaia-dataset-builder/
├── backend/     FastAPI + SQLAlchemy + SQLite
├── frontend/    Next.js 15 + TypeScript + TailwindCSS
└── storage/     raw / validated / rejected / exports
```

## Development (without Docker)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Monorepo, Docker, FastAPI, Next.js, SQLite, base UI |
| 2 | ✅ | Wikimedia + iNaturalist connectors, search grid, autocomplete |
| 3 | ✅ | Background image download, SHA256 + perceptual hash, thumbnails |
| 4 | ✅ | Export system (classification/YOLO/COCO/CSV), dedup detection, dashboard chart |
| 5 | ⏳ | Tests, optimizations, GBIF connector |
