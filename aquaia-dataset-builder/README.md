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

## Workflow

1. **Search** — enter a scientific name (e.g. `Ephemeroptera`) → images retrieved from Wikimedia & iNaturalist
2. **Validation Queue** — review images, press `V` (validate), `R` (reject), `D` (duplicate)
3. **Dataset Explorer** — browse validated images and taxonomy
4. **Export Center** — export as Classification / YOLO / COCO / CSV

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
