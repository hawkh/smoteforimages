"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { EpochData } from "@/lib/use-training-ws";

interface Props {
  data: EpochData[];
}

export default function TrainingChart({ data }: Props) {
  if (data.length === 0) return null;

  const chartData = data.map((d) => ({
    epoch: d.epoch,
    recon: d.recon_loss ?? null,
    g_loss: d.g_loss ?? null,
    d_loss: d.d_loss ?? null,
    fm_loss: d.fm_loss ?? null,
  }));

  const hasGan = chartData.some((d) => d.g_loss !== null);

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="epoch"
            tick={{ fill: "#71717a", fontSize: 11 }}
            stroke="#3f3f46"
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 11 }}
            stroke="#3f3f46"
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#a1a1aa" }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }}
          />
          <Line
            type="monotone"
            dataKey="recon"
            name="Recon Loss"
            stroke="#6366f1"
            dot={false}
            strokeWidth={2}
          />
          {hasGan && (
            <>
              <Line
                type="monotone"
                dataKey="g_loss"
                name="G Loss"
                stroke="#22c55e"
                dot={false}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="d_loss"
                name="D Loss"
                stroke="#f59e0b"
                dot={false}
                strokeWidth={1.5}
              />
              <Line
                type="monotone"
                dataKey="fm_loss"
                name="FM Loss"
                stroke="#ec4899"
                dot={false}
                strokeWidth={1.5}
              />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
