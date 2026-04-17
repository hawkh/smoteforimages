// API response types — mirrors api/models/responses.py

export interface ClassInfo {
  name: string;
  count: number;
}

export interface DatasetResponse {
  dataset_id: string;
  name: string;
  classes: ClassInfo[];
  total_images: number;
}

export interface DatasetDetailResponse extends DatasetResponse {
  sample_images: Record<string, string[]>; // class_name -> base64 thumbnails
}

export interface RunSummary {
  run_id: string;
  dataset_id: string;
  status: "idle" | "training" | "trained" | "generating" | "complete" | "error";
  config: Record<string, unknown>;
  created_at: string;
}

export interface ConfigureResponse {
  run_id: string;
  config_summary: Record<string, unknown>;
}

export interface TrainStartResponse {
  status: string;
  run_id: string;
  ws_url: string;
}

export interface TrainStatusResponse {
  run_id: string;
  status: string;
  phase: string | null;
  epoch: number | null;
  total_epochs: number | null;
  is_complete: boolean;
  metrics: Record<string, number> | null;
}

export interface GenerateResponse {
  run_id: string;
  n_generated: number;
  class_breakdown: Record<string, number>;
  output_dir: string;
}

export interface ImageResult {
  url: string;
  class_name: string;
  filename: string;
  is_synthetic: boolean;
}

export interface PaginatedImages {
  images: ImageResult[];
  total: number;
  page: number;
  per_page: number;
}

export interface QualityReport {
  run_id: string;
  metrics: Record<string, number>;
  diversity: Record<string, number> | null;
  per_class: Record<string, Record<string, number>> | null;
}

export interface PatentSection {
  id: string;
  title: string;
  content_md: string;
}

export interface EquationInfo {
  id: string;
  name: string;
  latex: string;
  description: string;
}

export interface ArchComponent {
  name: string;
  description: string;
  type: string;
  connections: string[];
}

export interface ArchPhase {
  name: string;
  epochs: string;
  losses: string[];
}

export interface ArchitectureResponse {
  components: ArchComponent[];
  phases: ArchPhase[];
}

// Pipeline config form
export interface PipelineConfig {
  dataset_id: string;
  image_size: number;
  embedding_dim: number;
  architecture: "resnet18" | "resnet50";
  pretrained: boolean;
  base_channels: number;
  use_self_attention: boolean;
  class_embed_dim: number;
  use_slerp: boolean;
  use_vmf: boolean;
  vmf_concentration_scale: number;
  k_neighbors: number;
  quality_metrics: string[];
}

// WS training event
export interface TrainingEvent {
  type: "epoch" | "complete" | "error" | "ping";
  data: Record<string, unknown>;
}
