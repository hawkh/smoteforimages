"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { FlaskConical, Play, Settings2, Zap } from "lucide-react";
import { listDatasets, configurePipeline } from "@/lib/api";
import type { DatasetResponse, PipelineConfig } from "@/lib/types";

const DEFAULTS: Omit<PipelineConfig, "dataset_id"> = {
  image_size: 64,
  embedding_dim: 512,
  architecture: "resnet18",
  pretrained: true,
  base_channels: 256,
  use_self_attention: true,
  class_embed_dim: 64,
  use_slerp: true,
  use_vmf: false,
  vmf_concentration_scale: 1.0,
  k_neighbors: 5,
  quality_metrics: ["mse", "psnr", "ssim"],
};

export default function PipelinePage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [cfg, setCfg] = useState(DEFAULTS);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDatasets().then((d) => {
      setDatasets(d);
      if (d.length > 0) setDatasetId(d[0].dataset_id);
    });
  }, []);

  async function handleCreate() {
    if (!datasetId) return;
    setCreating(true);
    setError(null);
    try {
      const res = await configurePipeline({ ...cfg, dataset_id: datasetId });
      router.push(`/pipeline/${res.run_id}`);
    } catch (e) {
      setError(String(e));
      setCreating(false);
    }
  }

  function Field({
    label,
    children,
  }: {
    label: string;
    children: React.ReactNode;
  }) {
    return (
      <label className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-zinc-400">{label}</span>
        {children}
      </label>
    );
  }

  const inputCls =
    "rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none";
  const selectCls = inputCls;

  return (
    <div className="p-8 max-w-3xl">
      <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
        <FlaskConical size={22} className="text-indigo-400" />
        New Pipeline Run
      </h1>
      <p className="mt-1 text-sm text-zinc-500">
        Configure encoder, decoder, and SMOTE parameters
      </p>

      {datasets.length === 0 ? (
        <div className="mt-8 rounded-xl border border-zinc-800 bg-zinc-900/50 p-8 text-center text-sm text-zinc-500">
          No datasets uploaded yet. Go to{" "}
          <a href="/datasets" className="text-indigo-400 hover:underline">
            Datasets
          </a>{" "}
          first.
        </div>
      ) : (
        <div className="mt-8 space-y-6">
          {/* Dataset selection */}
          <Field label="Dataset">
            <select
              className={selectCls}
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
            >
              {datasets.map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.name} ({d.total_images} images, {d.classes.length} classes)
                </option>
              ))}
            </select>
          </Field>

          {/* Core settings */}
          <div className="grid grid-cols-3 gap-4">
            <Field label="Image Size">
              <select
                className={selectCls}
                value={cfg.image_size}
                onChange={(e) => setCfg({ ...cfg, image_size: Number(e.target.value) })}
              >
                {[32, 64, 128, 256].map((s) => (
                  <option key={s} value={s}>{s}px</option>
                ))}
              </select>
            </Field>
            <Field label="Architecture">
              <select
                className={selectCls}
                value={cfg.architecture}
                onChange={(e) =>
                  setCfg({ ...cfg, architecture: e.target.value as "resnet18" | "resnet50" })
                }
              >
                <option value="resnet18">ResNet-18</option>
                <option value="resnet50">ResNet-50</option>
              </select>
            </Field>
            <Field label="Embedding Dim">
              <select
                className={selectCls}
                value={cfg.embedding_dim}
                onChange={(e) => setCfg({ ...cfg, embedding_dim: Number(e.target.value) })}
              >
                {[128, 256, 512, 1024].map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </Field>
          </div>

          {/* SMOTE mode */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Zap size={14} className="text-amber-400" />
              Oversampling Mode
            </h3>
            <div className="mt-3 flex gap-3">
              {[
                { label: "SLERP", desc: "Geodesic interpolation", key: "slerp" as const },
                { label: "vMF", desc: "Distributional sampling", key: "vmf" as const },
              ].map(({ label, desc, key }) => {
                const active =
                  key === "slerp" ? cfg.use_slerp && !cfg.use_vmf : cfg.use_vmf;
                return (
                  <button
                    key={key}
                    onClick={() =>
                      setCfg({
                        ...cfg,
                        use_slerp: key === "slerp",
                        use_vmf: key === "vmf",
                      })
                    }
                    className={`flex-1 rounded-lg border p-4 text-left transition-colors ${
                      active
                        ? "border-indigo-500 bg-indigo-500/10"
                        : "border-zinc-700 bg-zinc-800 hover:border-zinc-600"
                    }`}
                  >
                    <div className="text-sm font-medium">{label}</div>
                    <div className="mt-0.5 text-xs text-zinc-500">{desc}</div>
                  </button>
                );
              })}
            </div>
            {cfg.use_vmf && (
              <Field label="vMF Concentration Scale">
                <input
                  type="number"
                  className={inputCls + " mt-2 w-32"}
                  value={cfg.vmf_concentration_scale}
                  min={0.1}
                  max={10}
                  step={0.1}
                  onChange={(e) =>
                    setCfg({ ...cfg, vmf_concentration_scale: Number(e.target.value) })
                  }
                />
              </Field>
            )}
          </div>

          {/* Advanced toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <Settings2 size={12} />
            {showAdvanced ? "Hide" : "Show"} advanced options
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-3 gap-4 rounded-xl border border-zinc-800 bg-zinc-900/30 p-5">
              <Field label="Base Channels">
                <input
                  type="number"
                  className={inputCls}
                  value={cfg.base_channels}
                  min={64}
                  max={512}
                  step={64}
                  onChange={(e) => setCfg({ ...cfg, base_channels: Number(e.target.value) })}
                />
              </Field>
              <Field label="Self-Attention">
                <select
                  className={selectCls}
                  value={String(cfg.use_self_attention)}
                  onChange={(e) => setCfg({ ...cfg, use_self_attention: e.target.value === "true" })}
                >
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </Field>
              <Field label="Class Embed Dim">
                <input
                  type="number"
                  className={inputCls}
                  value={cfg.class_embed_dim}
                  min={16}
                  max={256}
                  step={16}
                  onChange={(e) => setCfg({ ...cfg, class_embed_dim: Number(e.target.value) })}
                />
              </Field>
              <Field label="K Neighbors">
                <input
                  type="number"
                  className={inputCls}
                  value={cfg.k_neighbors}
                  min={2}
                  max={20}
                  onChange={(e) => setCfg({ ...cfg, k_neighbors: Number(e.target.value) })}
                />
              </Field>
              <Field label="Pretrained Encoder">
                <select
                  className={selectCls}
                  value={String(cfg.pretrained)}
                  onChange={(e) => setCfg({ ...cfg, pretrained: e.target.value === "true" })}
                >
                  <option value="true">Yes (ImageNet)</option>
                  <option value="false">No</option>
                </select>
              </Field>
            </div>
          )}

          {/* Submit */}
          {error && (
            <p className="text-sm text-red-400">{error}</p>
          )}
          <button
            onClick={handleCreate}
            disabled={creating || !datasetId}
            className="flex items-center gap-2 rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-40"
          >
            <Play size={14} />
            {creating ? "Creating..." : "Create Pipeline Run"}
          </button>
        </div>
      )}
    </div>
  );
}
