import axios from "axios";
import type {
  DashboardStats,
  ImageRecord,
  PaginatedResponse,
  SearchQuery,
  Taxon,
  ExportJob,
} from "@/types";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api",
  timeout: 30000,
});

// Stats
export const getStats = () =>
  api.get<DashboardStats>("/stats").then((r) => r.data);

// Search
export const getSearchHistory = (limit = 20) =>
  api.get<SearchQuery[]>("/search", { params: { limit } }).then((r) => r.data);

export const runSearch = (query: string, sources: string[], limit = 50) =>
  api
    .post<ImageRecord[]>("/search/run", { query, sources, limit })
    .then((r) => r.data);

// Images
export const getImages = (params: {
  page?: number;
  size?: number;
  status?: string;
  source?: string;
  taxon_id?: number;
}) => api.get<PaginatedResponse<ImageRecord>>("/images", { params }).then((r) => r.data);

export const updateImageStatus = (
  id: number,
  status: string,
  notes?: string
) =>
  api
    .patch<ImageRecord>(`/images/${id}/status`, { status, notes })
    .then((r) => r.data);

// Taxonomy
export const getTaxons = (q?: string) =>
  api
    .get<Taxon[]>("/taxonomy", { params: q ? { q } : {} })
    .then((r) => r.data);

// Exports
export const getExports = () =>
  api.get<ExportJob[]>("/exports").then((r) => r.data);

export const createExport = (exportType: string, parameters?: object) =>
  api
    .post<ExportJob>("/exports", { export_type: exportType, parameters })
    .then((r) => r.data);
