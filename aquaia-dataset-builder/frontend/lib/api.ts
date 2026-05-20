import axios from "axios";
import type {
  DashboardStats,
  Dataset,
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

// Datasets
export const getDatasets = () =>
  api.get<Dataset[]>("/datasets").then((r) => r.data);

export const createDataset = (name: string, description?: string) =>
  api.post<Dataset>("/datasets", { name, description }).then((r) => r.data);

export const deleteDataset = (id: number) =>
  api.delete(`/datasets/${id}`);

export const addImageToDataset = (datasetId: number, imageId: number) =>
  api.post(`/datasets/${datasetId}/images/${imageId}`);

export const removeImageFromDataset = (datasetId: number, imageId: number) =>
  api.delete(`/datasets/${datasetId}/images/${imageId}`);

// Image import
export const importFromUrl = (url: string, scientific_name?: string) =>
  api
    .post<ImageRecord>("/images/import-url", { url, scientific_name })
    .then((r) => r.data);

export const cropImage = (imageId: number, blob: Blob) => {
  const form = new FormData();
  form.append("file", blob, "crop.jpg");
  return api
    .post<ImageRecord>(`/images/${imageId}/crop`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((r) => r.data);
};

export const uploadFiles = (files: File[], scientific_name?: string) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  if (scientific_name) form.append("scientific_name", scientific_name);
  return api
    .post<ImageRecord[]>("/images/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((r) => r.data);
};
