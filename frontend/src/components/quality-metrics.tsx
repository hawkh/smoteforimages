"use client";

import type { QualityReport } from "@/lib/types";

interface Props {
  report: QualityReport;
}

const METRIC_META: Record<string, { label: string; unit: string; good: "high" | "low" }> = {
  mse: { label: "Mean Squared Error", unit: "", good: "low" },
  psnr: { label: "Peak Signal-to-Noise", unit: "dB", good: "high" },
  ssim: { label: "Structural Similarity", unit: "", good: "high" },
  fid: { label: "Frechet Inception Distance", unit: "", good: "low" },
  lpips: { label: "Learned Perceptual", unit: "", good: "low" },
};

export default function QualityMetrics({ report }: Props) {
  const metrics = Object.entries(report.metrics);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
      <h3 className="text-sm font-semibold text-zinc-300 mb-4">Quality Assessment</h3>

      <div className="grid grid-cols-3 gap-3">
        {metrics.map(([key, value]) => {
          const meta = METRIC_META[key] ?? { label: key, unit: "", good: "low" };
          return (
            <div
              key={key}
              className="rounded-lg border border-zinc-800 bg-zinc-950 p-4"
            >
              <div className="text-[10px] text-zinc-500 uppercase tracking-wider">
                {meta.label}
              </div>
              <div className="mt-1.5 flex items-baseline gap-1">
                <span className="text-xl font-mono font-semibold text-white">
                  {typeof value === "number" ? value.toFixed(4) : value}
                </span>
                {meta.unit && (
                  <span className="text-xs text-zinc-500">{meta.unit}</span>
                )}
              </div>
              <div className="mt-1 text-[10px] text-zinc-600">
                {meta.good === "high" ? "Higher is better" : "Lower is better"}
              </div>
            </div>
          );
        })}
      </div>

      {/* Per-class breakdown */}
      {report.per_class && (
        <div className="mt-5">
          <h4 className="text-xs font-medium text-zinc-500 mb-2">Per-Class Breakdown</h4>
          <div className="overflow-hidden rounded-lg border border-zinc-800">
            <table className="w-full text-xs">
              <thead className="bg-zinc-900">
                <tr className="text-zinc-500">
                  <th className="px-3 py-2 text-left">Class</th>
                  {metrics.map(([key]) => (
                    <th key={key} className="px-3 py-2 text-right font-mono">{key.toUpperCase()}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {Object.entries(report.per_class).map(([cls, vals]) => (
                  <tr key={cls} className="hover:bg-zinc-800/50">
                    <td className="px-3 py-2 font-medium">{cls}</td>
                    {metrics.map(([key]) => (
                      <td key={key} className="px-3 py-2 text-right font-mono text-zinc-400">
                        {typeof vals[key] === "number" ? vals[key].toFixed(4) : "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
