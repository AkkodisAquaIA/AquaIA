from datetime import datetime
from typing import Optional
from sqlalchemy import Integer, String, Text, DateTime, ForeignKey, BigInteger, Table, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base


# ── Many-to-many: Dataset ↔ ImageRecord ────────────────────────────────────

dataset_images = Table(
    "dataset_images",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True),
    Column("image_id",   Integer, ForeignKey("image_records.id", ondelete="CASCADE"), primary_key=True),
)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    images: Mapped[list["ImageRecord"]] = relationship(
        "ImageRecord", secondary=dataset_images, back_populates="datasets"
    )


class Taxon(Base):
    __tablename__ = "taxons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    scientific_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    common_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    rank: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    parent_taxon_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("taxons.id"), nullable=True)
    reference_image_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("image_records.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    images: Mapped[list["ImageRecord"]] = relationship("ImageRecord", back_populates="taxon", foreign_keys="ImageRecord.taxon_id")


class ImageRecord(Base):
    __tablename__ = "image_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    taxon_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("taxons.id"), nullable=True)
    source_name: Mapped[str] = mapped_column(String(100))
    source_image_url: Mapped[str] = mapped_column(Text)
    source_page_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    sha256_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    perceptual_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    validated_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    taxon: Mapped[Optional["Taxon"]] = relationship("Taxon", back_populates="images")
    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset", secondary=dataset_images, back_populates="images"
    )


class SearchQuery(Base):
    __tablename__ = "search_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    query: Mapped[str] = mapped_column(String(500))
    filters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    export_type: Mapped[str] = mapped_column(String(50))
    parameters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
