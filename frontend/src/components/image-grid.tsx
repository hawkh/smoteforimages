"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { getResults } from "@/lib/api";
import type { PaginatedImages } from "@/lib/types";

interface Props {
  images: PaginatedImages;
  runId: string;
}

export default function ImageGrid({ images: initial, runId }: Props) {
  const [data, setData] = useState(initial);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<string | undefined>();

  const classes = [...new Set(initial.images.map((i) => i.class_name))];

  async function goToPage(page: number) {
    setLoading(true);
    try {
      const res = await getResults(runId, page, 24, filter);
      setData(res);
    } catch {
      // keep current
    }
    setLoading(false);
  }

  async function handleFilter(cls: string | undefined) {
    setFilter(cls);
    setLoading(true);
    try {
      const res = await getResults(runId, 1, 24, cls);
      setData(res);
    } catch {
      // keep current
    }
    setLoading(false);
  }

  const totalPages = Math.ceil(data.total / data.per_page);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-zinc-300">
          Synthetic Images ({data.total})
        </h3>
        <div className="flex gap-1.5">
          <button
            onClick={() => handleFilter(undefined)}
            className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
              !filter ? "bg-indigo-500/20 text-indigo-400" : "bg-zinc-800 text-zinc-500 hover:text-zinc-300"
            }`}
          >
            All
          </button>
          {classes.map((cls) => (
            <button
              key={cls}
              onClick={() => handleFilter(cls)}
              className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
                filter === cls ? "bg-indigo-500/20 text-indigo-400" : "bg-zinc-800 text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {cls}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex h-48 items-center justify-center text-sm text-zinc-500">
          Loading...
        </div>
      ) : (
        <div className="grid grid-cols-6 gap-2">
          {data.images.map((img) => (
            <div key={img.filename} className="group relative">
              <img
                src={img.url}
                alt={`${img.class_name} — ${img.filename}`}
                className="aspect-square w-full rounded-lg border border-zinc-700 object-cover"
              />
              <div className="absolute inset-x-0 bottom-0 rounded-b-lg bg-black/60 px-2 py-1 text-[10px] text-zinc-300 opacity-0 group-hover:opacity-100 transition-opacity">
                {img.class_name}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-4 flex items-center justify-center gap-2">
          <button
            onClick={() => goToPage(data.page - 1)}
            disabled={data.page <= 1}
            className="rounded-lg border border-zinc-700 p-1.5 text-zinc-400 hover:bg-zinc-800 disabled:opacity-30"
          >
            <ChevronLeft size={14} />
          </button>
          <span className="text-xs text-zinc-500">
            {data.page} / {totalPages}
          </span>
          <button
            onClick={() => goToPage(data.page + 1)}
            disabled={data.page >= totalPages}
            className="rounded-lg border border-zinc-700 p-1.5 text-zinc-400 hover:bg-zinc-800 disabled:opacity-30"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
