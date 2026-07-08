import axios from "axios";
import type {
  DashboardStats,
  Dataset,
  ImageRecord,
  PaginatedResponse,
  SearchHistory,
  Taxon,
  TaxonQueueItem,
  ExportJob,
  User,
} from "@/types";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api",
  timeout: 30000,
});

// Users / Workspaces
export const getUsers = () =>
  api.get<User[]>("/users").then((r) => r.data);

export const createUser = (display_name: string) =>
  api.post<User>("/users", { display_name }).then((r) => r.data);

export const updateUser = (id: number, display_name: string) =>
  api.patch<User>(`/users/${id}`, { display_name }).then((r) => r.data);

export const deleteUser = (id: number) =>
  api.delete(`/users/${id}`);

// Stats
export const getStats = (userId: number) =>
  api.get<DashboardStats>("/stats", { params: { user_id: userId } }).then((r) => r.data);

// Search
export const getSearchHistory = (userId: number, limit = 20) =>
  api.get<SearchHistory[]>("/search", { params: { user_id: userId, limit } }).then((r) => r.data);

export const runSearch = (userId: number, query: string, sources: string[], limit = 50) =>
  api
    .post<ImageRecord[]>("/search/run", { user_id: userId, query, sources, limit })
    .then((r) => r.data);

// Images
export const getImages = (
  userId: number,
  params: { page?: number; size?: number; status?: string; source?: string; taxon_id?: number }
) =>
  api
    .get<PaginatedResponse<ImageRecord>>("/images", { params: { user_id: userId, ...params } })
    .then((r) => r.data);

export const clearImages = (userId: number, status?: string, taxonId?: number) =>
  api.delete("/images", { params: { user_id: userId, ...(status ? { status } : {}), ...(taxonId ? { taxon_id: taxonId } : {}) } });

export const getTaxonQueue = (userId: number) =>
  api.get<TaxonQueueItem[]>("/taxonomy/queue", { params: { user_id: userId } }).then((r) => r.data);

export const updateImageStatus = (userId: number, id: number, status: string, notes?: string) =>
  api
    .patch<ImageRecord>(`/images/${id}/status`, { status, notes }, { params: { user_id: userId } })
    .then((r) => r.data);

export const cropImage = (
  userId: number,
  imageId: number,
  x: number,
  y: number,
  width: number,
  height: number
) =>
  api
    .post<ImageRecord>(`/images/${imageId}/crop`, { x, y, width, height }, { params: { user_id: userId } })
    .then((r) => r.data);

// Taxonomy
export const getTaxons = (userId: number, q?: string) =>
  api
    .get<Taxon[]>("/taxonomy", { params: { user_id: userId, ...(q ? { q } : {}) } })
    .then((r) => r.data);

export const setTaxonReference = (taxonId: number, userImageId: number | null, userId: number) =>
  api
    .patch<Taxon>(`/taxonomy/${taxonId}/reference`, { user_id: userId, user_image_id: userImageId })
    .then((r) => r.data);

// Exports
export const getExports = (userId: number) =>
  api.get<ExportJob[]>("/exports", { params: { user_id: userId } }).then((r) => r.data);

export const createExport = (userId: number, exportType: string, datasetId?: number, parameters?: object) =>
  api
    .post<ExportJob>("/exports", {
      user_id: userId,
      export_type: exportType,
      dataset_id: datasetId ?? null,
      parameters: parameters ?? null,
    })
    .then((r) => r.data);

// Datasets
export const getDatasets = (userId: number) =>
  api.get<Dataset[]>("/datasets", { params: { user_id: userId } }).then((r) => r.data);

export const createDataset = (userId: number, name: string, description?: string) =>
  api
    .post<Dataset>("/datasets", { name, description }, { params: { user_id: userId } })
    .then((r) => r.data);

export const deleteDataset = (userId: number, id: number) =>
  api.delete(`/datasets/${id}`, { params: { user_id: userId } });

export const getDatasetImages = (userId: number, datasetId: number) =>
  api
    .get<ImageRecord[]>(`/datasets/${datasetId}/images`, { params: { user_id: userId } })
    .then((r) => r.data);

export const addImageToDataset = (userId: number, datasetId: number, imageId: number) =>
  api.post(`/datasets/${datasetId}/images/${imageId}`, null, { params: { user_id: userId } });

export const getAssignedImages = (userId: number) =>
  api
    .get<Record<number, { dataset_id: number; dataset_name: string }>>("/datasets/assigned", {
      params: { user_id: userId },
    })
    .then((r) => r.data);

export const removeImageFromDataset = (userId: number, datasetId: number, imageId: number) =>
  api.delete(`/datasets/${datasetId}/images/${imageId}`, { params: { user_id: userId } });

// Image import
export const importFromUrl = (userId: number, url: string, scientific_name?: string) =>
  api
    .post<ImageRecord>("/images/import-url", { user_id: userId, url, scientific_name })
    .then((r) => r.data);

export const uploadFiles = (userId: number, files: File[], scientific_name?: string) => {
  const form = new FormData();
  form.append("user_id", String(userId));
  files.forEach((f) => form.append("files", f));
  if (scientific_name) form.append("scientific_name", scientific_name);
  return api
    .post<ImageRecord[]>("/images/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((r) => r.data);
};
