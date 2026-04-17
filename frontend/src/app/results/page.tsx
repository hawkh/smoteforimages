"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Sparkles, FlaskConical } from "lucide-react";
import { listRuns } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

export default function ResultsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listRuns()
      .then((r) => setRuns(r.filter((run) => run.status === "complete")))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
        <Sparkles size={22} className="text-emerald-400" />
        Results
      </h1>
      <p className="mt-1 text-sm text-zinc-500">
        Completed pipeline runs with generated images
      </p>

      {loading ? (
        <div className="mt-8 text-sm text-zinc-500">Loading...</div>
      ) : runs.length === 0 ? (
        <div className="mt-8 rounded-xl border border-zinc-800 bg-zinc-900/50 p-8 text-center">
          <FlaskConical size={32} className="mx-auto text-zinc-600" />
          <p className="mt-3 text-sm text-zinc-500">
            No completed runs yet.{" "}
            <Link href="/pipeline" className="text-indigo-400 hover:underline">
              Start a pipeline
            </Link>
          </p>
        </div>
      ) : (
        <div className="mt-6 grid grid-cols-2 gap-4">
          {runs.map((r) => {
            const cfg = r.config as Record<string, unknown>;
            return (
              <Link
                key={r.run_id}
                href={`/pipeline/${r.run_id}`}
                className="group rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 hover:border-zinc-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm text-zinc-300">
                    {r.run_id.slice(0, 12)}
                  </span>
                  <span className="rounded-full bg-emerald-500/10 px-2.5 py-0.5 text-xs text-emerald-400">
                    complete
                  </span>
                </div>
                <div className="mt-2 text-xs text-zinc-500">
                  {r.dataset_id} &middot; {cfg.architecture as string} &middot;{" "}
                  {cfg.image_size as number}px &middot;{" "}
                  {cfg.use_slerp ? "SLERP" : "vMF"}
                </div>
                <div className="mt-2 text-xs text-zinc-600">{r.created_at}</div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
