"use client";

import { useEffect, useState, useCallback } from "react";
import { Upload, Trash2, FolderOpen, ImageIcon } from "lucide-react";
import { listDatasets, uploadDataset, deleteDataset, getDataset } from "@/lib/api";
import type { DatasetResponse, DatasetDetailResponse } from "@/lib/types";

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [selected, setSelected] = useState<DatasetDetailResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const refresh = useCallback(() => {
    listDatasets().then(setDatasets).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleFiles(items: FileList | File[]) {
    const files: File[] = [];
    const paths: string[] = [];

    for (const f of Array.from(items)) {
      // webkitRelativePath gives us class_name/filename structure
      const relPath = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
      files.push(f);
      paths.push(relPath);
    }

    if (files.length === 0) return;

    setUploading(true);
    try {
      await uploadDataset(files, paths);
      refresh();
    } catch (e) {
      console.error("Upload failed:", e);
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
  }

  async function handleDelete(id: string) {
    await deleteDataset(id);
    if (selected?.dataset_id === id) setSelected(null);
    refresh();
  }

  async function handleSelect(id: string) {
    try {
      const detail = await getDataset(id);
      setSelected(detail);
    } catch {
      setSelected(null);
    }
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold tracking-tight">Datasets</h1>
      <p className="mt-1 text-sm text-zinc-500">
        Upload image folders organized as <code className="text-xs font-mono text-zinc-400">class_name/image.jpg</code>
      </p>

      {/* Upload zone */}
      <div
        className={`mt-6 flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 transition-colors ${
          dragActive
            ? "border-indigo-500 bg-indigo-500/5"
            : "border-zinc-700 bg-zinc-900/30 hover:border-zinc-600"
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <Upload size={28} className="text-zinc-500" />
        <p className="mt-3 text-sm text-zinc-400">
          Drag & drop a folder, or{" "}
          <label className="cursor-pointer text-indigo-400 hover:underline">
            browse
            <input
              type="file"
              className="hidden"
              multiple
              /* @ts-expect-error webkitdirectory */
              webkitdirectory=""
              onChange={(e) => e.target.files && handleFiles(e.target.files)}
            />
          </label>
        </p>
        {uploading && (
          <p className="mt-2 text-xs text-amber-400 animate-pulse">Uploading...</p>
        )}
      </div>

      <div className="mt-8 grid grid-cols-3 gap-6">
        {/* Dataset list */}
        <div className="col-span-1">
          <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Uploaded ({datasets.length})
          </h2>
          {datasets.length === 0 ? (
            <p className="text-sm text-zinc-600">No datasets yet.</p>
          ) : (
            <div className="flex flex-col gap-2">
              {datasets.map((d) => (
                <button
                  key={d.dataset_id}
                  onClick={() => handleSelect(d.dataset_id)}
                  className={`flex items-center justify-between rounded-lg border px-4 py-3 text-left text-sm transition-colors ${
                    selected?.dataset_id === d.dataset_id
                      ? "border-indigo-500 bg-indigo-500/10"
                      : "border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800"
                  }`}
                >
                  <div>
                    <div className="flex items-center gap-2 font-medium">
                      <FolderOpen size={14} className="text-zinc-500" />
                      {d.name}
                    </div>
                    <div className="mt-0.5 text-xs text-zinc-500">
                      {d.classes.length} classes &middot; {d.total_images} images
                    </div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDelete(d.dataset_id); }}
                    className="text-zinc-600 hover:text-red-400 transition-colors"
                    aria-label={`Delete dataset ${d.name}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Detail panel */}
        <div className="col-span-2">
          {selected ? (
            <div>
              <h2 className="text-lg font-semibold">{selected.name}</h2>
              <p className="text-sm text-zinc-500">
                {selected.classes.length} classes &middot; {selected.total_images} images
              </p>

              <div className="mt-4 flex flex-wrap gap-3">
                {selected.classes.map((c) => (
                  <span
                    key={c.name}
                    className="rounded-full border border-zinc-700 bg-zinc-800 px-3 py-1 text-xs font-mono"
                  >
                    {c.name} ({c.count})
                  </span>
                ))}
              </div>

              {/* Thumbnails */}
              {Object.entries(selected.sample_images).map(([cls, thumbs]) => (
                <div key={cls} className="mt-6">
                  <h3 className="text-sm font-medium text-zinc-400 mb-2">{cls}</h3>
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {thumbs.map((b64, i) => (
                      <img
                        key={i}
                        src={`data:image/jpeg;base64,${b64}`}
                        alt={`${cls} sample ${i + 1}`}
                        className="h-20 w-20 rounded-lg border border-zinc-700 object-cover"
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex h-64 flex-col items-center justify-center text-zinc-600">
              <ImageIcon size={32} />
              <p className="mt-2 text-sm">Select a dataset to preview</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
