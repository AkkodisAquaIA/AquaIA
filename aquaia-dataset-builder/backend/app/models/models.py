from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class User(Base):
    """Workspace / user — optional password protection."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def is_protected(self) -> bool:
        return self.password_hash is not None

    user_images: Mapped[list["UserImage"]] = relationship("UserImage", back_populates="user", cascade="all, delete-orphan")
    datasets: Mapped[list["Dataset"]] = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")
    exports: Mapped[list["ExportJob"]] = relationship("ExportJob", back_populates="user", cascade="all, delete-orphan")
    search_history: Mapped[list["SearchHistory"]] = relationship("SearchHistory", back_populates="user", cascade="all, delete-orphan")
    settings: Mapped[Optional["UserSettings"]] = relationship("UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")
    taxon_references: Mapped[list["UserTaxonReference"]] = relationship("UserTaxonReference", back_populates="user", cascade="all, delete-orphan")


class Taxon(Base):
    """Global taxonomy — shared by all users."""

    __tablename__ = "taxons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    scientific_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    common_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    rank: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    parent_taxon_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("taxons.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    image_assets: Mapped[list["ImageAsset"]] = relationship("ImageAsset", back_populates="taxon")


class ImageAsset(Base):
    """Global image repository — one record per physical file, shared by all users."""

    __tablename__ = "image_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    taxon_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("taxons.id"), nullable=True)
    source_name: Mapped[str] = mapped_column(String(100))
    source_image_url: Mapped[str] = mapped_column(Text, unique=True, index=True)
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    taxon: Mapped[Optional[Taxon]] = relationship("Taxon", back_populates="image_assets")
    user_images: Mapped[list["UserImage"]] = relationship("UserImage", back_populates="asset")


class UserImage(Base):
    """Per-user decision on an ImageAsset (status, notes, dataset membership)."""

    __tablename__ = "user_images"
    __table_args__ = (UniqueConstraint("user_id", "image_asset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    image_asset_id: Mapped[int] = mapped_column(Integer, ForeignKey("image_assets.id", ondelete="CASCADE"), index=True)
    taxon_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("taxons.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="user_images")
    asset: Mapped[ImageAsset] = relationship("ImageAsset", back_populates="user_images")
    taxon: Mapped[Optional[Taxon]] = relationship("Taxon")
    dataset_memberships: Mapped[list["DatasetImage"]] = relationship("DatasetImage", back_populates="user_image", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="datasets")
    dataset_images: Mapped[list["DatasetImage"]] = relationship("DatasetImage", back_populates="dataset", cascade="all, delete-orphan")


class DatasetImage(Base):
    __tablename__ = "dataset_images"
    __table_args__ = (UniqueConstraint("dataset_id", "user_image_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    user_image_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_images.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="dataset_images")
    user_image: Mapped[UserImage] = relationship("UserImage", back_populates="dataset_memberships")


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    export_type: Mapped[str] = mapped_column(String(50))
    parameters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="exports")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    settings_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="settings")


class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(String(500))
    sources: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="search_history")


class UserTaxonReference(Base):
    """Per-user reference image for a taxon shown in the Validation Queue."""

    __tablename__ = "user_taxon_references"
    __table_args__ = (UniqueConstraint("user_id", "taxon_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    taxon_id: Mapped[int] = mapped_column(Integer, ForeignKey("taxons.id", ondelete="CASCADE"), index=True)
    user_image_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_images.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="taxon_references")
