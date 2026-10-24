/** API client for the Dendrux dashboard. */

import type {
  LLMCallsResponse,
  LLMCallsResult,
  RunDetailResponse,
  RunListResponse,
} from "./types";

/**
 * Resolve the API base URL.
 *
 * When served standalone (`dendrux dashboard`), the page is at `/` and API at `/api`.
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

/**
 * Get the auth token from sessionStorage.
 * If a 401 is received, prompt the user for a token.
 */
function getAuthToken(): string | null {
  return sessionStorage.getItem("dendrux_auth_token");
}

function setAuthToken(token: string): void {
  sessionStorage.setItem("dendrux_auth_token", token);
}

async function get<T>(path: string): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  let res = await fetch(`${BASE}${path}`, { headers });

  // If 401, prompt for token and retry once
  if (res.status === 401) {
    const input = prompt("Dashboard auth token required:");
    if (input) {
      setAuthToken(input);
      headers["Authorization"] = `Bearer ${input}`;
      res = await fetch(`${BASE}${path}`, { headers });
    }
  }

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

/** Pages the server serves per request are bounded; load at most this many pages. */
const LLM_CALL_MAX_PAGES = 5;

/** Load a run's LLM calls page by page, stopping at a bound and saying so. */
export async function fetchLLMCalls(runId: string): Promise<LLMCallsResult> {
  const calls: LLMCallsResult["llm_calls"] = [];
  let offset = 0;
  for (let page = 0; page < LLM_CALL_MAX_PAGES; page++) {
    const res = await get<LLMCallsResponse>(`/runs/${runId}/llm-calls?offset=${offset}`);
    calls.push(...res.llm_calls);
    if (!res.has_more) return { llm_calls: calls, incomplete: false };
    if (res.next_offset == null) break;
    offset = res.next_offset;
  }
  return { llm_calls: calls, incomplete: true };
}

export async function fetchHealth(): Promise<{ status: string }> {
  return get<{ status: string }>("/health");
}
