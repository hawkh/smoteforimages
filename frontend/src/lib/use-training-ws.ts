"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { TrainingEvent } from "./types";

export interface EpochData {
  epoch: number;
  total_epochs: number;
  phase: string;
  recon_loss?: number;
  g_loss?: number;
  d_loss?: number;
  fm_loss?: number;
  [key: string]: unknown;
}

export function useTrainingWs(runId: string | null) {
  const [events, setEvents] = useState<EpochData[]>([]);
  const [status, setStatus] = useState<"idle" | "connected" | "complete" | "error">("idle");
  const [lastError, setLastError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!runId) return;

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/training/${runId}`);
    wsRef.current = ws;

    ws.onopen = () => setStatus("connected");

    ws.onmessage = (e) => {
      const msg: TrainingEvent = JSON.parse(e.data);
      if (msg.type === "epoch") {
        setEvents((prev) => [...prev, msg.data as unknown as EpochData]);
      } else if (msg.type === "complete") {
        setStatus("complete");
      } else if (msg.type === "error") {
        setStatus("error");
        setLastError(String(msg.data?.message ?? "Unknown error"));
      }
      // ping events ignored
    };

    ws.onerror = () => {
      setStatus("error");
      setLastError("WebSocket connection error");
    };

    ws.onclose = () => {
      if (status === "connected") setStatus("complete");
    };
  }, [runId, status]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  const reset = useCallback(() => {
    setEvents([]);
    setStatus("idle");
    setLastError(null);
  }, []);

  return { events, status, lastError, reset };
}
