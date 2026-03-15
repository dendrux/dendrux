/** API client for the Dendrite dashboard. */

import type { RunDetailResponse, RunListResponse } from "./types";

/**
 * Resolve the API base URL.
 *
 * When served standalone (`dendrite dashboard`), the page is at `/` and API at `/api`.
 * When mounted as a sub-app (e.g. `/dashboard`), the page is at `/dashboard/`
 * and API at `/dashboard/api`.
 *
 * We derive the base from the HTML document's own URL.
 */
function getApiBase(): string {
  // In dev (Vite proxy), always use /api
  if (import.meta.env.DEV) return "/api";

  // In production, resolve relative to current page path
  const base = document.baseURI || window.location.href;
  const url = new URL("api", base);
  return url.pathname.replace(/\/$/, "");
}

const BASE = getApiBase();

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchRuns(params?: {
  limit?: number;
  offset?: number;
  status?: string;
  agent?: string;
}): Promise<RunListResponse> {
  const search = new URLSearchParams();
  if (params?.limit) search.set("limit", String(params.limit));
  if (params?.offset) search.set("offset", String(params.offset));
  if (params?.status) search.set("status", params.status);
  if (params?.agent) search.set("agent", params.agent);
  const qs = search.toString();
  return get<RunListResponse>(`/runs${qs ? `?${qs}` : ""}`);
}

export async function fetchRunDetail(runId: string): Promise<RunDetailResponse> {
  return get<RunDetailResponse>(`/runs/${runId}`);
}

export async function fetchHealth(): Promise<{ status: string }> {
  return get<{ status: string }>("/health");
}
