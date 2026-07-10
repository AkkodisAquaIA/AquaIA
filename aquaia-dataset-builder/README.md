# ADIAB — AquaIA Dataset Builder

Professional AI dataset platform for aquatic macro-invertebrate identification.

## Quick start

All commands below must be run from the `aquaia-dataset-builder/` directory:

```bash
cd aquaia-dataset-builder
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## Production deployment (shared server)

To deploy ADIAB on a server so the whole team can access it, run from the `aquaia-dataset-builder/` directory:

```bash
cd aquaia-dataset-builder
docker compose -f docker-compose.prod.yml up -d --build
```

The app is then accessible at `http://<server-ip>` (port 80). No other port needs to be open.

**What's different from dev:**

| | Dev (`docker-compose.yml`) | Prod (`docker-compose.prod.yml`) |
|--|--|--|
| Entry point | Frontend :3000, API :8000 (separate ports) | Nginx :80 (single entry point) |
| Frontend build | `next dev` (hot reload) | `next build` + `next start` (optimised) |
| Backend | `uvicorn --reload` | `uvicorn --workers 2` (no hot reload) |
| Storage | bind-mounted `./storage/` | same |

Storage data (`storage/raw`, `storage/thumbnails`, `storage/exports`, `adiab.db`) is preserved on the server between restarts — it is never removed by `docker compose down`.

To update ADIAB after a git pull, re-run:
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

---

## Docker commands (dev)

> Run all commands from the `aquaia-dataset-builder/` directory.

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

Only images with status `validated` appear here. A dataset groups **species folders** (taxon folders), not individual images.

**How to create a dataset and add species:**

1. Click the **Datasets tab** → **+ New dataset** → type a name (e.g. `Macroinvertebrates v1`) → **Create**
2. Click the **Validated tab** → validated images are shown grouped by species folder
3. On each species folder header, click **Add to dataset** → a dropdown lists your datasets
4. Choose an existing dataset (a checkmark appears if the folder is already there) — or click **New dataset…** to create one and assign the folder in one action
5. The dataset card updates immediately: species badges and image count are shown

You can add as many species folders as you want to the same dataset. A dataset contains:
- **One species** (e.g. `Glossiphonia_complanata` only)
- **Or multiple species** (e.g. `Glossiphonia_complanata` + `Branchiobdella` + `Orectochilus_villosus`)

**Viewing and editing a dataset:**
- Click a dataset card to open its detail view → species are shown as collapsible sub-folders
- Each sub-folder has a **Remove species** button that removes all images of that taxon from the dataset (the images stay in your Validated library)
- The pencil icon on a dataset card lets you rename it inline (Enter to confirm, Escape to cancel)

**Deleting images from the Validated library:**
- Hover over an individual image → red trash icon → confirm
- Use the folder trash icon to delete all images of a species at once (with confirmation)

| Tab | Content |
|-----|---------|
| **Datasets** | Named collections. Each shows its species (indigo badges) and image count. Create, rename, browse, delete datasets here. |
| **Validated** | All validated images grouped by species folder. Assign folders to datasets or delete images here. |
| **Taxons** | Taxonomy built automatically from searches (scientific name, common name, rank). |

### 4. Export Center — export for AI training

- Choose a format:

| Format | Description |
|--------|-------------|
| **Classification** | One folder per taxon — standard for `torchvision.ImageFolder`, Keras `flow_from_directory` |
| **YOLO** | `images/<taxon>/` sub-folders + `data.yaml` class list |
| **COCO JSON** | `images/` flat folder + `instances_validated.json` in COCO format |
| **CSV** | `metadata.csv` flat table with image path, taxon, license, author… |

- Click **Create export job** → a zip is generated in the background
- Click **Download** once the job status shows `done` (page auto-refreshes while running)

#### Source & Licence attribution files (SL files)

Every export zip automatically includes **SL attribution files** — one per species folder — so the source and licence of every image is always documented alongside the data.

**Naming convention:** `SL_<FOLDER_NAME_UPPERCASE>.csv`

Examples:
- `Glossiphonia_complanata/SL_GLOSSIPHONIA_COMPLANATA.csv`
- `Branchiobdella/SL_BRANCHIOBDELLA.csv`

**File format (CSV, UTF-8):**

| Column | Content |
|--------|---------|
| `source_url` | Full URL of the original image page (e.g. `https://www.inaturalist.org/photos/12345`) |
| `author` | Author / photographer name as provided by the source |
| `license` | Licence identifier (e.g. `CC BY 4.0`, `CC0`, `CC BY-NC 4.0`) |

**One row per image** in the folder, in the same order as the image files.

**Export format coverage:**

| Export format | SL file location |
|---------------|-----------------|
| Classification | `<taxon>/SL_<TAXON>.csv` — one file per species folder |
| YOLO | `images/<taxon>/SL_<TAXON>.csv` — one file per species folder |
| COCO JSON | `SL_SOURCES.csv` at the root (flat, all images) |
| CSV | `SL_SOURCES.csv` at the root (flat, all images) |

**Why?** Images sourced from Wikimedia Commons, iNaturalist and GBIF are almost all under Creative Commons licences. These licences require attribution — the creator's name must be cited and the licence link included in any derived work (including AI training datasets). The SL file makes it trivial to fulfil this legal obligation without having to re-look up metadata after the fact.

> Tip: if you plan to publish your training dataset or share it with partners, keep the SL files in the archive — they serve as the attribution record.

### 5. Settings — platform configuration

| Setting | Description |
|---------|-------------|
| **Default crop dimensions** | Size of the pre-initialized crop rectangle when opening the crop tool. Presets: 224 / 256 / 320 / 416 / 512 / **640** (default) × same height. Custom W × H also supported. Persisted in browser localStorage. |
| **Reference images** | One representative image per species, used as a visual anchor in the Validation Queue. Shows all defined references with a remove button. |

> Tip: set crop dimensions to match your model's input size (e.g. 640×640 for YOLOv8, 224×224 for ResNet/EfficientNet) so every crop is already the right size without post-processing.

**Reference image — Validation Queue detail panel:**

The bottom of the right panel shows the reference image for the selected species:
- If a reference is already set → thumbnail is shown with a gold **★ Référence** badge; click **Changer** to replace it with the current image
- If no reference is set → a dashed **"Définir comme référence"** button lets you set the current image in one click
- The separator between the selected image and the reference image is **draggable** — drag it up or down to resize both sections

## Storage

Images are stored locally under `aquaia-dataset-builder/storage/` (never committed to git):

```
storage/
├── raw/              ← shared downloaded images (named by asset id, e.g. 42.jpg)
├── thumbnails/       ← shared auto-generated 256×256 previews
├── exports/
│   ├── user_1/       ← workspace-specific exports
│   ├── user_2/
│   └── user_3/
└── adiab.db          ← SQLite database
```

## Architecture

```
aquaia-dataset-builder/
├── backend/     FastAPI + SQLAlchemy + SQLite
├── frontend/    Next.js 15 + TypeScript + TailwindCSS
└── storage/     raw / validated / rejected / exports
```

## Architecture: shared image assets, isolated workspaces

ADIAB uses a **hybrid two-layer architecture** that separates physical image storage from per-user decisions. The goal is to avoid downloading the same image multiple times while keeping each workspace's validation queue, datasets, exports and statistics completely private.

### Global shared layer

| Element | Description |
|---------|-------------|
| `ImageAsset` | One record per unique image URL — the canonical image identity |
| `Taxon` | Shared taxonomy (scientific name, common name, rank) |
| Source metadata | Provider, source URL, author, license |
| Raw image files | `storage/raw/{asset_id}.jpg` — written once, never duplicated |
| Thumbnails | `storage/thumbnails/{asset_id}.jpg` — generated once |

### User-isolated layer

| Element | Description |
|---------|-------------|
| `User` / Workspace | Lightweight identity — display name only, no password |
| `UserImage` | Per-user decision on an `ImageAsset` (status, notes) |
| Validation status | `pending` / `validated` / `rejected` / `duplicate` — per workspace |
| Datasets | Collections of validated images — private to each workspace |
| Dataset memberships | Which images belong to which dataset — per workspace |
| Exports | Generated zip files — per workspace, stored under `storage/exports/user_{id}/` |
| Settings | Crop dimensions, preferences — per workspace |
| Search history | Past queries and result counts — per workspace |
| Dashboard statistics | Image counts, taxon count, recent searches — per workspace |

### Data flow diagram

```
Internet sources
(Wikimedia / iNaturalist / GBIF)
        ↓
   Image search
        ↓
Shared ImageAsset repository
(one physical file per unique URL)
        ↓
┌──────────────────────┬──────────────────────┐
│ Workspace A          │ Workspace B          │
│ - own queue          │ - own queue          │
│ - own statuses       │ - own statuses       │
│ - own datasets       │ - own datasets       │
│ - own exports        │ - own exports        │
└──────────────────────┴──────────────────────┘
```

### Workflow

1. A user selects or creates a workspace in the top-right selector.
2. The user runs a search, e.g. `Ephemeroptera`.
3. The backend fetches images from external sources and stores each **`ImageAsset` only once** globally (upsert by URL).
4. For the active workspace, the backend creates **`UserImage`** records with status `pending`.
5. Each workspace validates, rejects or marks images as duplicate **independently**.
6. Datasets are created and managed only inside the current workspace.
7. Exports are generated only from the current workspace's datasets or validated images.
8. Dashboard statistics are calculated per workspace.

### Concrete example

- Workspace A validates image #42.
- Workspace B rejects image #42.
- Both decisions are valid — the `ImageAsset` is shared but each `UserImage` carries an independent status.
- The physical file `storage/raw/42.jpg` exists **only once** on disk.

### Database model overview

| Model | Layer | Description |
|-------|-------|-------------|
| `User` | Isolated | A workspace — display name, created date. No passwords. |
| `Taxon` | Shared | Scientific name, common name, rank, optional parent. |
| `ImageAsset` | Shared | One record per unique image URL. Holds file path, hash, dimensions, metadata. |
| `UserImage` | Isolated | Links a `User` to an `ImageAsset`. Holds status, notes, taxon override. |
| `Dataset` | Isolated | Named collection owned by a workspace. |
| `DatasetImage` | Isolated | Join table between `Dataset` and `UserImage`. |
| `ExportJob` | Isolated | Export task — format, status, output path. Scoped to a workspace. |
| `UserSettings` | Isolated | JSON settings blob per workspace (crop size, preferences). |
| `SearchHistory` | Isolated | Past search queries and result counts per workspace. |
| `UserTaxonReference` | Isolated | Per-workspace reference image for each taxon (shown in Validation Queue). |

### Storage layout

```
storage/
├── raw/              ← shared downloaded images  (named by asset id, e.g. 42.jpg)
├── thumbnails/       ← shared thumbnails         (named by asset id)
├── exports/
│   ├── user_1/       ← workspace-specific exports
│   ├── user_2/
│   └── user_3/
└── adiab.db          ← SQLite database
```

## Why this architecture?

- **No duplicate downloads** — the same image fetched by multiple workspaces is downloaded and stored only once.
- **Saves disk space** — `storage/raw/` grows with unique images, not with the number of users × images.
- **Independent validation** — two users can make different decisions on the same image; neither affects the other.
- **Private datasets** — each workspace builds its own curated collections without interference.
- **Ready for authentication** — adding OAuth or password login later only requires linking an auth identity to an existing `User` row.
- **Ready for collaborative annotation** — the shared `ImageAsset` layer makes it straightforward to compare annotations across workspaces.
- **Scalable for a team** — adding a new workspace is a single `INSERT` into the `users` table; all infrastructure already handles multi-user access.

## Workspace isolation rules

- A workspace only sees its **own validation queue** (`UserImage` rows where `user_id` matches).
- A workspace only sees its **own datasets** and dataset memberships.
- A workspace only sees its **own exports** and can only download its own zip files.
- A workspace has its **own settings** (crop size, preferences).
- **Dashboard statistics** are calculated exclusively from the current workspace's `UserImage` rows.
- **Image files and thumbnails** are globally shared — never per-user copies.
- One image can have **different statuses** (validated / rejected / pending) in different workspaces simultaneously.

## Example workflow with two users

1. **Marie** creates workspace `Marie` in the selector.
2. Marie searches `Ephemeroptera` → 50 images appear in her Validation Queue as `pending`.
3. Marie validates 10 images and creates dataset `Ephemeroptera v1`.
4. **Paul** creates workspace `Paul` in the selector.
5. Paul's Validation Queue is **empty** — he has no `UserImage` rows yet.
6. Paul searches `Ephemeroptera` → the backend reuses the existing `ImageAsset` records (no re-download) and creates Paul's own `UserImage` rows with status `pending`.
7. Paul rejects several images that Marie validated — both decisions coexist independently.
8. Marie exports `Ephemeroptera v1` as YOLO; Paul exports his own selection as CSV. Both get separate zip files under `storage/exports/user_1/` and `storage/exports/user_2/`.

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
