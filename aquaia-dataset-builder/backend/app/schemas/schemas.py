from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── User ──────────────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    display_name: str = Field(..., max_length=100)
    password: str = Field(..., min_length=6, max_length=128, pattern=r"^\d+$")


class UserUpdate(BaseModel):
    display_name: str = Field(..., max_length=100)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    display_name: str
    is_protected: bool = False
    created_at: datetime


class UserWithToken(UserRead):
    access_token: str


# ── Taxon ──────────────────────────────────────────────────────────────────────


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
    reference_image_id: Optional[int] = None  # UserImage.id for the current user
    created_at: datetime


# ── ImageRecord (flat view of UserImage + ImageAsset) ─────────────────────────
#
# The `id` field is UserImage.id.  File-serving endpoints also take UserImage.id.


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


class ImageRecordRead(BaseModel):
    """Flat view: id = UserImage.id, asset fields merged in."""

    model_config = ConfigDict(from_attributes=True)

    id: int  # UserImage.id
    asset_id: int  # ImageAsset.id
    user_id: int

    # Asset fields
    source_name: str
    source_image_url: str
    source_page_url: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    local_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    sha256_hash: Optional[str] = None

    # UserImage fields
    taxon_id: Optional[int] = None
    status: str
    notes: Optional[str] = None
    created_at: datetime
    validated_at: Optional[datetime] = None
    taxon: Optional[TaxonRead] = None


def image_record_read(ui, ref_image_id: Optional[int] = None) -> "ImageRecordRead":
    """Build an ImageRecordRead from a UserImage ORM instance."""
    asset = ui.asset
    taxon_data = None
    if ui.taxon:
        taxon_data = TaxonRead(
            id=ui.taxon.id,
            scientific_name=ui.taxon.scientific_name,
            common_name=ui.taxon.common_name,
            rank=ui.taxon.rank,
            parent_taxon_id=ui.taxon.parent_taxon_id,
            reference_image_id=ref_image_id,
            created_at=ui.taxon.created_at,
        )
    return ImageRecordRead(
        id=ui.id,
        asset_id=asset.id,
        user_id=ui.user_id,
        source_name=asset.source_name,
        source_image_url=asset.source_image_url,
        source_page_url=asset.source_page_url,
        author=asset.author,
        license=asset.license,
        local_path=asset.local_path,
        thumbnail_path=asset.thumbnail_path,
        width=asset.width,
        height=asset.height,
        file_size=asset.file_size,
        sha256_hash=asset.sha256_hash,
        taxon_id=ui.taxon_id,
        status=ui.status,
        notes=ui.notes,
        created_at=ui.created_at,
        validated_at=ui.validated_at,
        taxon=taxon_data,
    )


# ── Dataset ────────────────────────────────────────────────────────────────────


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    image_count: int = 0


# ── Import ─────────────────────────────────────────────────────────────────────


ALLOWED_SOURCES = {"wikimedia", "inaturalist", "gbif"}


class ImportUrlRequest(BaseModel):
    url: str = Field(..., max_length=2048)
    scientific_name: Optional[str] = Field(None, max_length=300)
    user_id: int


# ── Search ─────────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500)
    sources: list[str] = Field(default=["wikimedia", "inaturalist", "gbif"], max_length=10)
    limit: int = Field(50, ge=1, le=200)
    taxon_id: Optional[int] = None
    user_id: int


class SearchHistoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    query: str
    sources: Optional[str] = None
    result_count: int
    created_at: datetime


# ── Export ─────────────────────────────────────────────────────────────────────


class ExportRequest(BaseModel):
    export_type: Literal["classification", "yolo", "coco", "csv"]
    user_id: int
    dataset_id: Optional[int] = None
    parameters: Optional[dict[str, Any]] = None


class ExportJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None
    export_type: str
    output_path: Optional[str] = None
    status: str
    created_at: datetime


# ── Stats ──────────────────────────────────────────────────────────────────────


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
    recent_searches: list[SearchHistoryRead]


# ── Pagination ─────────────────────────────────────────────────────────────────


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int
    size: int
    pages: int
