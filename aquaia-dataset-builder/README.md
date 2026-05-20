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

## Usage

### 1. Search — retrieve images

- Click **Search** in the sidebar
- Type a scientific name: `Ephemeroptera`, `Baetis rhodani`, `Plecoptera`…
- Select sources (Wikimedia Commons, iNaturalist)
- Click **Search** → images are fetched and saved to the database automatically

### 2. Validation Queue — review images

- Click **Validation Queue**
- Click an image to select it
- Use buttons or keyboard shortcuts:
  - `V` — Validate ✅
  - `R` — Reject ❌
  - `D` — Mark as duplicate
  - `Space` — Deselect
- Use the filter tabs to browse Pending / Validated / Rejected / Duplicates

### 3. Dataset Explorer — browse your dataset

- See all validated images in a grid
- Browse the taxonomy created automatically from your searches

### 4. Export Center — export for AI training

- Choose a format: **Classification** (folders) · **YOLO** · **COCO JSON** · **CSV**
- Click **Create export job**

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
| 2 | ⏳ | Wikimedia + iNaturalist connectors, search grid |
| 3 | ⏳ | Image persistence, local storage, download |
| 4 | ⏳ | Export system, duplicate detection, dashboard charts |
| 5 | ⏳ | UI polish, animations, tests, optimizations |
