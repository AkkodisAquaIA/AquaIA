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

Three ways to add images:

**By scientific name** (Search tab)
- Type a scientific name: `Ephemeroptera`, `Baetis rhodani`, `Plecoptera`…
- Select one or more sources: **Wikimedia Commons**, **iNaturalist**, **GBIF**
- Set **Max results** to control how many images are fetched per source: 10 / 25 / **50** (default) / 100 / 200
- Click **Search** → images are fetched from each selected source and saved as `pending`

> Tip: start with 50 (default) to get a representative sample without flooding your validation queue. Use 100–200 for rare species where results are sparse.

**By URL** (Add by URL tab)
- Paste a direct image URL
- Optionally set the scientific name for automatic taxon assignment
- The image is downloaded and processed in the background

**From your computer** (Upload tab)
- Drag and drop image files (jpg, png, webp…) or click to browse
- Optionally set a scientific name
- SHA256 exact-duplicate check runs automatically on upload

### 2. Validation Queue — review images

Every image retrieved by Search lands here with status `pending`. Nothing enters your dataset until you explicitly validate it.

**How to review:**
- Click an image to select it — the detail panel opens on the right
- After each action the panel **automatically moves to the next image** — no need to re-click
- Click the preview image (or hover for the zoom icon) to open a **fullscreen lightbox** for closer inspection

**Lightbox crop tool:**
- In the lightbox, click **Crop** to enter crop mode
- A selection rectangle is pre-positioned and sized to your default crop dimensions (see Settings)
- Drag to reposition, handles to resize — then click **Apply crop**
- The cropped image replaces the original: dimensions, hash and thumbnail are all updated
- Press `Escape` or click **Cancel** to exit without saving

- Assign a status using the buttons or keyboard shortcuts:

| Status | Button | Key | Effect |
|--------|--------|-----|--------|
| **Pending** | — | — | Initial state of every newly fetched image — waiting for your review |
| **Validated** | Validate | `V` | Image is included in exports and Dataset Explorer |
| **Rejected** | Reject | `R` | Image is excluded — bad quality, wrong species, off-topic |
| **Duplicate** | Duplicate | `D` | Image is a near-copy of another one already in the database |
| **Later** | Later | — | Deferred for a second pass — stays visible in the Later tab |
| Deselect | — | `Space` | Closes the detail panel without changing the status |

**Filter tabs** let you browse each category: Pending / Validated / Rejected / Duplicates / Later

**Automatic duplicate detection:** when an image is downloaded, the backend computes a perceptual hash (dhash). If two images have a Hamming distance ≤ 8 (visually near-identical), the newer one is automatically marked `duplicate` — you don't need to do it manually. The `Duplicate` button is for cases the algorithm misses (e.g. same insect, slightly different crop or angle).

### 3. Dataset Explorer — organise validated images

Only images with status `validated` appear here.

- **Datasets tab**: create named collections (e.g. "Ephemeroptera training v1"), see image count and creation date per dataset
- **Validated tab**: browse all validated images as a grid
- **Taxons tab**: view the taxonomy built automatically from searches (scientific name, common name, rank)

### 4. Export Center — export for AI training

- Choose a format:

| Format | Description |
|--------|-------------|
| **Classification** | One folder per taxon — standard for `torchvision.ImageFolder`, Keras `flow_from_directory` |
| **YOLO** | `images/` + `labels/` folders with `.txt` annotation files |
| **COCO JSON** | Single `annotations.json` in COCO format |
| **CSV** | Flat table with image path, taxon, license, author… |

- Click **Create export job** → a zip is generated in the background
- Click **Download** once the job status shows `done` (page auto-refreshes while running)

### 5. Settings — platform configuration

| Setting | Description |
|---------|-------------|
| **Default crop dimensions** | Size of the pre-initialized crop rectangle when opening the crop tool. Presets: 224 / 256 / 320 / 416 / 512 / **640** (default) × same height. Custom W × H also supported. Persisted in browser localStorage. |

> Tip: set crop dimensions to match your model's input size (e.g. 640×640 for YOLOv8, 224×224 for ResNet/EfficientNet) so every crop is already the right size without post-processing.

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
