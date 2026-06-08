export type PanelId =
  | "dashboard"
  | "search"
  | "validation"
  | "dataset"
  | "export"
  | "settings";

export interface User {
  id: number;
  username: string;
  display_name: string;
  created_at: string;
}

export interface Taxon {
  id: number;
  scientific_name: string;
  common_name: string | null;
  rank: string | null;
  parent_taxon_id: number | null;
  reference_image_id: number | null;
  created_at: string;
}

export interface ImageRecord {
  id: number;         // UserImage.id
  asset_id: number;   // ImageAsset.id
  user_id: number;
  taxon_id: number | null;
  source_name: string;
  source_image_url: string;
  source_page_url: string | null;
  author: string | null;
  license: string | null;
  local_path: string | null;
  thumbnail_path: string | null;
  width: number | null;
  height: number | null;
  file_size: number | null;
  sha256_hash: string | null;
  status: "pending" | "validated" | "rejected" | "duplicate" | "review_later";
  notes: string | null;
  created_at: string;
  validated_at: string | null;
  taxon: Taxon | null;
}

export interface SearchHistory {
  id: number;
  query: string;
  sources: string | null;
  result_count: number;
  created_at: string;
}

export interface ExportJob {
  id: number;
  user_id: number;
  dataset_id: number | null;
  export_type: string;
  output_path: string | null;
  status: string;
  created_at: string;
}

export interface Dataset {
  id: number;
  user_id: number;
  name: string;
  description: string | null;
  created_at: string;
  image_count: number;
}

export interface DashboardStats {
  total_images: number;
  pending: number;
  validated: number;
  rejected: number;
  duplicates: number;
  downloaded: number;
  total_taxons: number;
  total_datasets: number;
  total_exports: number;
  recent_searches: SearchHistory[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}
