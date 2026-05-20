export type PanelId =
  | "dashboard"
  | "search"
  | "validation"
  | "dataset"
  | "export"
  | "settings";

export interface Taxon {
  id: number;
  scientific_name: string;
  common_name: string | null;
  rank: string | null;
  parent_taxon_id: number | null;
  created_at: string;
}

export interface ImageRecord {
  id: number;
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
  validated_by: string | null;
  taxon: Taxon | null;
}

export interface SearchQuery {
  id: number;
  query: string;
  source: string | null;
  result_count: number;
  created_at: string;
}

export interface ExportJob {
  id: number;
  export_type: string;
  output_path: string | null;
  status: string;
  created_at: string;
}

export interface DashboardStats {
  total_images: number;
  pending: number;
  validated: number;
  rejected: number;
  duplicates: number;
  total_taxons: number;
  total_exports: number;
  recent_searches: SearchQuery[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}
