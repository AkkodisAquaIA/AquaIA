from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict


# ── Taxon ──────────────────────────────────────────────────────────────────

class TaxonBase(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    rank: Optional[str] = None
    parent_taxon_id: Optional[int] = None


class TaxonCreate(TaxonBase):
    pass


class TaxonRead(TaxonBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


# ── ImageRecord ────────────────────────────────────────────────────────────

class ImageRecordBase(BaseModel):
    taxon_id: Optional[int] = None
    source_name: str
    source_image_url: str
    source_page_url: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    status: str = "pending"
    notes: Optional[str] = None


class ImageRecordCreate(ImageRecordBase):
    pass


class ImageStatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None
    validated_by: Optional[str] = None


class ImageRecordRead(ImageRecordBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    local_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    sha256_hash: Optional[str] = None
    created_at: datetime
    validated_at: Optional[datetime] = None
    validated_by: Optional[str] = None
    taxon: Optional[TaxonRead] = None


# ── Dataset ────────────────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    image_count: int = 0


# ── Import ─────────────────────────────────────────────────────────────────

class ImportUrlRequest(BaseModel):
    url: str
    scientific_name: Optional[str] = None


# ── Search ─────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    sources: list[str] = ["wikimedia", "inaturalist", "gbif"]
    limit: int = 50
    taxon_id: Optional[int] = None


class SearchQueryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    query: str
    source: Optional[str] = None
    result_count: int
    created_at: datetime


# ── Export ─────────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    export_type: str
    parameters: Optional[dict[str, Any]] = None


class ExportJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    export_type: str
    output_path: Optional[str] = None
    status: str
    created_at: datetime


# ── Stats ──────────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_images: int
    pending: int
    validated: int
    rejected: int
    duplicates: int
    downloaded: int = 0
    total_taxons: int
    total_datasets: int = 0
    total_exports: int
    recent_searches: list[SearchQueryRead]


# ── Pagination ─────────────────────────────────────────────────────────────

class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int
    size: int
    pages: int
