"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Database,
  FlaskConical,
  LayoutDashboard,
  FileText,
  Sparkles,
} from "lucide-react";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/pipeline", label: "Pipeline", icon: FlaskConical },
  { href: "/results", label: "Results", icon: Sparkles },
  { href: "/docs", label: "Patent Docs", icon: FileText },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950 px-3 py-6">
      <div className="mb-8 px-3">
        <h1 className="text-sm font-bold tracking-wider text-indigo-400 uppercase">
          SMOTE Studio
        </h1>
        <p className="mt-0.5 text-[11px] text-zinc-500">Image Synthesis Lab</p>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors ${
                active
                  ? "bg-indigo-500/10 text-indigo-400 font-medium"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              }`}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto px-3 text-[10px] text-zinc-600">
        vMF-SMOTE + WGAN-GP
      </div>
    </aside>
  );
}
