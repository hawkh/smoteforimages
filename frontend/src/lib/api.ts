// API client for the SMOTE synthesis backend

import type {
  DatasetResponse,
  DatasetDetailResponse,
  PipelineConfig,
  ConfigureResponse,
  TrainStartResponse,
  TrainStatusResponse,
  GenerateResponse,
  PaginatedImages,
  QualityReport,
  RunSummary,
  PatentSection,
  EquationInfo,
  ArchitectureResponse,
} from "./types";

const BASE = "";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// Datasets
export const listDatasets = () => json<DatasetResponse[]>("/api/datasets");
export const getDataset = (id: string) => json<DatasetDetailResponse>(`/api/datasets/${id}`);
export const deleteDataset = (id: string) =>
  fetch(`/api/datasets/${id}`, { method: "DELETE" });

export async function uploadDataset(files: File[], paths: string[], name?: string) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  paths.forEach((p) => form.append("paths", p));
  if (name) form.append("name", name);
  return json<DatasetResponse>("/api/datasets/upload", { method: "POST", body: form });
}

// Pipeline
export const listRuns = () => json<RunSummary[]>("/api/pipeline/runs");
export const getRun = (id: string) => json<RunSummary>(`/api/pipeline/runs/${id}`);

export const configurePipeline = (cfg: PipelineConfig) =>
  json<ConfigureResponse>("/api/pipeline/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });

export const startTraining = (runId: string, opts: { epochs: number; batch_size: number; learning_rate: number }) =>
  json<TrainStartResponse>(`/api/pipeline/train/${runId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });

export const getTrainStatus = (runId: string) =>
  json<TrainStatusResponse>(`/api/pipeline/status/${runId}`);

export const stopTraining = (runId: string) =>
  fetch(`/api/pipeline/stop/${runId}`, { method: "POST" });

export const generateImages = (runId: string, nSamples?: number) =>
  json<GenerateResponse>(`/api/pipeline/generate/${runId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n_samples: nSamples }),
  });

export const getResults = (runId: string, page = 1, perPage = 24, classFilter?: string) => {
  const params = new URLSearchParams({ page: String(page), per_page: String(perPage) });
  if (classFilter) params.set("class_filter", classFilter);
  return json<PaginatedImages>(`/api/pipeline/generate/${runId}/results?${params}`);
};

// Quality
export const evaluateQuality = (runId: string, metrics = ["mse", "psnr", "ssim"]) =>
  json<QualityReport>(`/api/quality/evaluate/${runId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ metrics, n_eval_samples: 100 }),
  });

export const getQualityReport = (runId: string) =>
  json<QualityReport>(`/api/quality/${runId}/report`);

// Docs
export const getPatentSections = () => json<PatentSection[]>("/api/docs/patent");
export const getEquations = () => json<EquationInfo[]>("/api/docs/equations");
export const getArchitecture = () => json<ArchitectureResponse>("/api/docs/architecture");
