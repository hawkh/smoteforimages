"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Database, FlaskConical, Sparkles, Activity } from "lucide-react";
import { listDatasets, listRuns } from "@/lib/api";
import type { DatasetResponse, RunSummary } from "@/lib/types";

export default function Dashboard() {
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      listDatasets().catch(() => []),
      listRuns().catch(() => []),
    ]).then(([d, r]) => {
      setDatasets(d);
      setRuns(r);
      setLoading(false);
    });
  }, []);

  const statusColor: Record<string, string> = {
    idle: "text-zinc-500",
    training: "text-amber-400",
    trained: "text-blue-400",
    generating: "text-purple-400",
    complete: "text-emerald-400",
    error: "text-red-400",
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
      <p className="mt-1 text-sm text-zinc-500">
        SMOTE Image Synthesis — class-conditional generation via SLERP/vMF
      </p>

      {/* Stats */}
      <div className="mt-8 grid grid-cols-4 gap-4">
        {[
          { label: "Datasets", value: datasets.length, icon: Database, color: "text-indigo-400" },
          { label: "Total Images", value: datasets.reduce((s, d) => s + d.total_images, 0), icon: Sparkles, color: "text-emerald-400" },
          { label: "Pipeline Runs", value: runs.length, icon: FlaskConical, color: "text-amber-400" },
          { label: "Completed", value: runs.filter((r) => r.status === "complete").length, icon: Activity, color: "text-blue-400" },
        ].map(({ label, value, icon: Icon, color }) => (
          <div key={label} className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
            <div className="flex items-center gap-2">
              <Icon size={16} className={color} />
              <span className="text-xs text-zinc-500 uppercase tracking-wider">{label}</span>
            </div>
            <div className="mt-2 text-2xl font-semibold font-mono">
              {loading ? "—" : value}
            </div>
          </div>
        ))}
      </div>

      {/* Recent runs */}
      <div className="mt-10">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Recent Runs</h2>
          <Link
            href="/pipeline"
            className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            New Run &rarr;
          </Link>
        </div>

        {loading ? (
          <div className="mt-4 text-sm text-zinc-500">Loading...</div>
        ) : runs.length === 0 ? (
          <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-8 text-center">
            <FlaskConical size={32} className="mx-auto text-zinc-600" />
            <p className="mt-3 text-sm text-zinc-500">
              No pipeline runs yet.{" "}
              <Link href="/pipeline" className="text-indigo-400 hover:underline">
                Start one
              </Link>
            </p>
          </div>
        ) : (
          <div className="mt-4 overflow-hidden rounded-xl border border-zinc-800">
            <table className="w-full text-sm">
              <thead className="bg-zinc-900/80">
                <tr className="text-left text-xs text-zinc-500 uppercase tracking-wider">
                  <th className="px-4 py-3">Run ID</th>
                  <th className="px-4 py-3">Dataset</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Created</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {runs.map((r) => (
                  <tr key={r.run_id} className="hover:bg-zinc-800/50 transition-colors">
                    <td className="px-4 py-3 font-mono text-xs">{r.run_id.slice(0, 8)}</td>
                    <td className="px-4 py-3">{r.dataset_id}</td>
                    <td className={`px-4 py-3 font-medium ${statusColor[r.status] ?? "text-zinc-400"}`}>
                      {r.status}
                    </td>
                    <td className="px-4 py-3 text-zinc-500">{r.created_at}</td>
                    <td className="px-4 py-3">
                      <Link
                        href={`/pipeline/${r.run_id}`}
                        className="text-indigo-400 hover:text-indigo-300 text-xs"
                      >
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
