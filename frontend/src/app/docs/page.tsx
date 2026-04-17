"use client";

import { useEffect, useState } from "react";
import { FileText, Atom, Network } from "lucide-react";
import EquationBlock from "@/components/equation-block";
import { getPatentSections, getEquations, getArchitecture } from "@/lib/api";
import type { PatentSection, EquationInfo, ArchitectureResponse } from "@/lib/types";

type Tab = "patent" | "equations" | "architecture";

export default function DocsPage() {
  const [tab, setTab] = useState<Tab>("equations");
  const [sections, setSections] = useState<PatentSection[]>([]);
  const [equations, setEquations] = useState<EquationInfo[]>([]);
  const [arch, setArch] = useState<ArchitectureResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getEquations().catch(() => []),
      getArchitecture().catch(() => null),
      getPatentSections().catch(() => []),
    ]).then(([eq, ar, sec]) => {
      setEquations(eq);
      setArch(ar);
      setSections(sec);
      setLoading(false);
    });
  }, []);

  const tabs: { id: Tab; label: string; icon: typeof FileText }[] = [
    { id: "equations", label: "Equations", icon: Atom },
    { id: "architecture", label: "Architecture", icon: Network },
    { id: "patent", label: "Patent Disclosure", icon: FileText },
  ];

  const typeColors: Record<string, string> = {
    encoder: "border-blue-500 bg-blue-500/10 text-blue-300",
    oversampler: "border-amber-500 bg-amber-500/10 text-amber-300",
    decoder: "border-emerald-500 bg-emerald-500/10 text-emerald-300",
    discriminator: "border-red-500 bg-red-500/10 text-red-300",
    controller: "border-purple-500 bg-purple-500/10 text-purple-300",
    ema: "border-zinc-500 bg-zinc-500/10 text-zinc-300",
  };

  return (
    <div className="p-8 max-w-5xl">
      <h1 className="text-2xl font-bold tracking-tight">Patent Documentation</h1>
      <p className="mt-1 text-sm text-zinc-500">
        Mathematical foundations and system architecture
      </p>

      {/* Tab bar */}
      <div className="mt-6 flex gap-1 border-b border-zinc-800 pb-px">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 rounded-t-lg px-4 py-2.5 text-sm transition-colors ${
              tab === id
                ? "bg-zinc-800 text-white font-medium"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="mt-8 text-sm text-zinc-500">Loading...</div>
      ) : (
        <div className="mt-6">
          {/* Equations */}
          {tab === "equations" && (
            <div className="space-y-4">
              {equations.map((eq) => (
                <div
                  key={eq.id}
                  className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5"
                >
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-indigo-500/20 px-2 py-0.5 text-[10px] font-mono text-indigo-400 uppercase">
                      {eq.id}
                    </span>
                    <h3 className="text-sm font-semibold">{eq.name}</h3>
                  </div>
                  <div className="mt-3 rounded-lg bg-zinc-950 p-4">
                    <EquationBlock latex={eq.latex} />
                  </div>
                  <p className="mt-3 text-xs text-zinc-500">{eq.description}</p>
                </div>
              ))}
            </div>
          )}

          {/* Architecture */}
          {tab === "architecture" && arch && (
            <div>
              {/* Components */}
              <h2 className="text-lg font-semibold mb-4">System Components</h2>
              <div className="grid grid-cols-2 gap-3">
                {arch.components.map((c) => (
                  <div
                    key={c.name}
                    className={`rounded-xl border p-4 ${typeColors[c.type] ?? "border-zinc-700 bg-zinc-900/50"}`}
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold">{c.name}</h3>
                      <span className="rounded-full bg-black/30 px-2 py-0.5 text-[10px] uppercase font-mono">
                        {c.type}
                      </span>
                    </div>
                    <p className="mt-1.5 text-xs opacity-80">{c.description}</p>
                    {c.connections.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {c.connections.map((conn) => (
                          <span
                            key={conn}
                            className="rounded bg-black/20 px-1.5 py-0.5 text-[10px] font-mono"
                          >
                            &rarr; {conn}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Training Phases */}
              <h2 className="text-lg font-semibold mt-8 mb-4">Training Phases</h2>
              <div className="flex gap-4">
                {arch.phases.map((p, i) => (
                  <div
                    key={p.name}
                    className="flex-1 rounded-xl border border-zinc-800 bg-zinc-900/50 p-5"
                  >
                    <div className="flex items-center gap-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-indigo-500/20 text-xs font-bold text-indigo-400">
                        {i + 1}
                      </span>
                      <h3 className="text-sm font-semibold">{p.name}</h3>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500 font-mono">
                      Epochs: {p.epochs}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {p.losses.map((l) => (
                        <span
                          key={l}
                          className="rounded-full border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-[10px] font-mono"
                        >
                          {l}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Patent sections */}
          {tab === "patent" && (
            <div>
              {/* TOC */}
              <div className="mb-6 rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">
                  Contents
                </h3>
                <div className="flex flex-wrap gap-2">
                  {sections.map((s) => (
                    <a
                      key={s.id}
                      href={`#${s.id}`}
                      className="text-xs text-indigo-400 hover:text-indigo-300"
                    >
                      {s.title}
                    </a>
                  ))}
                </div>
              </div>

              {/* Sections */}
              <div className="space-y-6">
                {sections.map((s) => (
                  <div
                    key={s.id}
                    id={s.id}
                    className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6"
                  >
                    <h2 className="text-lg font-semibold">{s.title}</h2>
                    <div className="mt-3 prose prose-invert prose-sm max-w-none text-zinc-400 whitespace-pre-wrap font-mono text-xs leading-relaxed">
                      {s.content_md}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
