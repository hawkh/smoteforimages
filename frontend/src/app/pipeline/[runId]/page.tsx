"use client";

import { use, useEffect, useState, useCallback } from "react";
import {
  Play,
  Square,
  Sparkles,
  BarChart3,
  Download,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import TrainingChart from "@/components/training-chart";
import ImageGrid from "@/components/image-grid";
import QualityMetrics from "@/components/quality-metrics";
import { useTrainingWs } from "@/lib/use-training-ws";
import {
  getRun,
  startTraining,
  stopTraining,
  generateImages,
  getResults,
  evaluateQuality,
} from "@/lib/api";
import type {
  RunSummary,
  GenerateResponse,
  PaginatedImages,
  QualityReport,
} from "@/lib/types";

export default function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = use(params);
  const [run, setRun] = useState<RunSummary | null>(null);
  const [phase, setPhase] = useState<"config" | "training" | "trained" | "generating" | "results">("config");
  const [wsActive, setWsActive] = useState(false);
  const { events, status: wsStatus, lastError } = useTrainingWs(wsActive ? runId : null);

  // Training params
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.0002);

  // Generation results
  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);
  const [images, setImages] = useState<PaginatedImages | null>(null);
  const [quality, setQuality] = useState<QualityReport | null>(null);
  const [loading, setLoading] = useState("");

  const refreshRun = useCallback(() => {
    getRun(runId).then((r) => {
      setRun(r);
      if (r.status === "training") {
        setPhase("training");
        setWsActive(true);
      } else if (r.status === "trained") {
        setPhase("trained");
      } else if (r.status === "complete") {
        setPhase("results");
      }
    });
  }, [runId]);

  useEffect(() => {
    refreshRun();
  }, [refreshRun]);

  // Watch WS status
  useEffect(() => {
    if (wsStatus === "complete") {
      setPhase("trained");
      setWsActive(false);
    } else if (wsStatus === "error") {
      setWsActive(false);
    }
  }, [wsStatus]);

  async function handleTrain() {
    setLoading("train");
    try {
      await startTraining(runId, { epochs, batch_size: batchSize, learning_rate: lr });
      setPhase("training");
      setWsActive(true);
    } catch (e) {
      console.error(e);
    }
    setLoading("");
  }

  async function handleStop() {
    await stopTraining(runId);
    setWsActive(false);
    setPhase("trained");
  }

  async function handleGenerate() {
    setLoading("generate");
    try {
      const res = await generateImages(runId);
      setGenResult(res);
      const imgs = await getResults(runId);
      setImages(imgs);
      setPhase("results");
    } catch (e) {
      console.error(e);
    }
    setLoading("");
  }

  async function handleEvaluate() {
    setLoading("evaluate");
    try {
      const q = await evaluateQuality(runId);
      setQuality(q);
    } catch (e) {
      console.error(e);
    }
    setLoading("");
  }

  const lastEvent = events[events.length - 1];
  const progress = lastEvent
    ? Math.round((lastEvent.epoch / lastEvent.total_epochs) * 100)
    : 0;

  const inputCls =
    "rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none w-full";

  return (
    <div className="p-8 max-w-5xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-bold tracking-tight font-mono">
          {runId.slice(0, 12)}...
        </h1>
        {run && (
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
              run.status === "complete"
                ? "bg-emerald-500/10 text-emerald-400"
                : run.status === "error"
                ? "bg-red-500/10 text-red-400"
                : run.status === "training"
                ? "bg-amber-500/10 text-amber-400"
                : "bg-zinc-700/50 text-zinc-400"
            }`}
          >
            {run.status}
          </span>
        )}
      </div>

      {run && (
        <p className="mt-1 text-sm text-zinc-500">
          Dataset: {run.dataset_id} &middot;{" "}
          {(run.config as Record<string, unknown>).architecture as string} &middot;{" "}
          {(run.config as Record<string, unknown>).image_size as number}px &middot;{" "}
          {(run.config as Record<string, unknown>).use_slerp ? "SLERP" : "vMF"}
        </p>
      )}

      {/* Step indicator */}
      <div className="mt-6 flex items-center gap-2 text-xs">
        {["Configure", "Train", "Generate", "Evaluate"].map((step, i) => {
          const stepIdx = { config: 0, training: 1, trained: 2, generating: 2, results: 3 }[phase] ?? 0;
          const done = i < stepIdx;
          const active = i === stepIdx;
          return (
            <div key={step} className="flex items-center gap-2">
              {i > 0 && <div className={`h-px w-8 ${done || active ? "bg-indigo-500" : "bg-zinc-700"}`} />}
              <div
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 ${
                  done
                    ? "bg-indigo-500/10 text-indigo-400"
                    : active
                    ? "bg-zinc-700 text-white"
                    : "bg-zinc-800/50 text-zinc-600"
                }`}
              >
                {done ? <CheckCircle2 size={12} /> : <span className="font-mono">{i + 1}</span>}
                {step}
              </div>
            </div>
          );
        })}
      </div>

      {/* Phase: Config / Train controls */}
      {(phase === "config" || phase === "trained") && (
        <div className="mt-8 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
          <h2 className="text-sm font-semibold text-zinc-300">
            {phase === "config" ? "Training Parameters" : "Re-train or Generate"}
          </h2>
          <div className="mt-4 grid grid-cols-3 gap-4">
            <label className="flex flex-col gap-1.5">
              <span className="text-xs text-zinc-500">Epochs</span>
              <input
                type="number"
                className={inputCls}
                value={epochs}
                min={1}
                max={1000}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-xs text-zinc-500">Batch Size</span>
              <input
                type="number"
                className={inputCls}
                value={batchSize}
                min={4}
                max={256}
                onChange={(e) => setBatchSize(Number(e.target.value))}
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-xs text-zinc-500">Learning Rate</span>
              <input
                type="number"
                className={inputCls}
                value={lr}
                step={0.0001}
                min={0.00001}
                max={0.1}
                onChange={(e) => setLr(Number(e.target.value))}
              />
            </label>
          </div>
          <div className="mt-5 flex gap-3">
            <button
              onClick={handleTrain}
              disabled={loading === "train"}
              className="flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-40"
            >
              {loading === "train" ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              Start Training
            </button>
            {phase === "trained" && (
              <button
                onClick={handleGenerate}
                disabled={loading === "generate"}
                className="flex items-center gap-2 rounded-lg bg-emerald-600 px-5 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
              >
                {loading === "generate" ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                Generate Images
              </button>
            )}
          </div>
        </div>
      )}

      {/* Phase: Training */}
      {phase === "training" && (
        <div className="mt-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Loader2 size={16} className="animate-spin text-amber-400" />
              <span className="text-sm font-medium">
                Training — Epoch {lastEvent?.epoch ?? 0}/{lastEvent?.total_epochs ?? epochs}
              </span>
              {lastEvent?.phase && (
                <span className="rounded-full bg-zinc-800 px-2.5 py-0.5 text-xs text-zinc-400">
                  {lastEvent.phase}
                </span>
              )}
            </div>
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 rounded-lg bg-red-600/20 px-3 py-1.5 text-xs text-red-400 hover:bg-red-600/30"
            >
              <Square size={12} />
              Stop
            </button>
          </div>

          {/* Progress bar */}
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-indigo-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-1 text-right text-[11px] text-zinc-600">{progress}%</p>

          {/* Loss chart */}
          <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <TrainingChart data={events} />
          </div>

          {/* Live metrics */}
          {lastEvent && (
            <div className="mt-4 grid grid-cols-4 gap-3">
              {[
                { label: "Recon Loss", value: lastEvent.recon_loss, color: "text-indigo-400" },
                { label: "G Loss", value: lastEvent.g_loss, color: "text-emerald-400" },
                { label: "D Loss", value: lastEvent.d_loss, color: "text-amber-400" },
                { label: "FM Loss", value: lastEvent.fm_loss, color: "text-pink-400" },
              ]
                .filter((m) => m.value != null)
                .map((m) => (
                  <div key={m.label} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
                    <div className="text-[10px] text-zinc-500 uppercase tracking-wider">{m.label}</div>
                    <div className={`mt-1 text-lg font-mono font-semibold ${m.color}`}>
                      {typeof m.value === "number" ? m.value.toFixed(4) : "—"}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}

      {/* WS error */}
      {lastError && (
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-800 bg-red-900/20 px-4 py-3 text-sm text-red-400">
          <AlertCircle size={14} />
          {lastError}
        </div>
      )}

      {/* Phase: Results */}
      {phase === "results" && (
        <div className="mt-8 space-y-6">
          {/* Generation summary */}
          {genResult && (
            <div className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
              <div>
                <p className="text-sm font-medium">
                  Generated {genResult.n_generated} synthetic images
                </p>
                <div className="mt-1 flex gap-2">
                  {Object.entries(genResult.class_breakdown).map(([cls, count]) => (
                    <span key={cls} className="rounded-full bg-zinc-800 px-2.5 py-0.5 text-xs font-mono">
                      {cls}: {count}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleEvaluate}
                  disabled={loading === "evaluate"}
                  className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-4 py-2 text-sm hover:bg-zinc-800 disabled:opacity-40"
                >
                  {loading === "evaluate" ? <Loader2 size={14} className="animate-spin" /> : <BarChart3 size={14} />}
                  Evaluate
                </button>
                <a
                  href={`/api/pipeline/generate/${runId}/download`}
                  className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-4 py-2 text-sm hover:bg-zinc-800"
                >
                  <Download size={14} />
                  ZIP
                </a>
              </div>
            </div>
          )}

          {/* Quality metrics */}
          {quality && <QualityMetrics report={quality} />}

          {/* Training chart (persisted) */}
          {events.length > 0 && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <h3 className="mb-3 text-sm font-semibold text-zinc-400">Training History</h3>
              <TrainingChart data={events} />
            </div>
          )}

          {/* Image gallery */}
          {images && <ImageGrid images={images} runId={runId} />}
        </div>
      )}
    </div>
  );
}
