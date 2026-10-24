/**
 * Behavioural tests for bounded reads: the real API client pages against a fake
 * server, and the real components render what the user would see.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { fetchLLMCalls } from "@/lib/api";
import { useDashboardStore } from "@/lib/store";
import { PayloadInspector, ProviderEmptyState } from "@/components/PayloadInspector";
import { DelegationPanel } from "@/components/DelegationPanel";
import { RunDetail } from "@/pages/RunDetail";
import type {
  DelegationInfo,
  LLMCallNode,
  LLMCallsResult,
  RunDetailResponse,
} from "@/lib/types";

// zustand's server render reads its *initial* state, so selection cannot be set from a
// test; replace the store with a plain mutable one that the components read as usual.
vi.mock("@/lib/store", () => {
  const state = {
    selectedNode: null as unknown,
    selectedIteration: null as number | null,
    clearSelection: () => {},
    selectNode: () => {},
  };
  const useDashboardStore = Object.assign(
    (selector?: (s: typeof state) => unknown) => (selector ? selector(state) : state),
    {
      getState: () => state,
      setState: (patch: Partial<typeof state>) => void Object.assign(state, patch),
    },
  );
  return { useDashboardStore };
});

const PAGE = 200; // the server's page size for /llm-calls

/** A fake dashboard API serving `total` LLM calls the way the real endpoint pages them. */
function fakeServer(total: number) {
  const requests: number[] = [];
  const fetchMock = vi.fn(async (input: string | URL) => {
    const url = new URL(String(input), "http://localhost");
    const offset = Number(url.searchParams.get("offset") ?? 0);
    requests.push(offset);
    const rows = Array.from({ length: Math.max(0, Math.min(PAGE, total - offset)) }, (_, i) => ({
      id: `call-${offset + i}`,
      iteration_index: offset + i + 1,
      semantic_request: { marker: `request-${offset + i + 1}` },
      semantic_response: { marker: `response-${offset + i + 1}` },
      provider_request: null,
      provider_response: null,
    }));
    const hasMore = offset + PAGE < total;
    return new Response(
      JSON.stringify({
        llm_calls: rows,
        limit: PAGE,
        offset,
        has_more: hasMore,
        next_offset: hasMore ? offset + PAGE : null,
        capped: false,
      }),
      { status: 200 },
    );
  });
  return { requests, fetchMock };
}

beforeEach(() => {
  const store = new Map<string, string>();
  vi.stubGlobal("sessionStorage", {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
  });
  useDashboardStore.setState({ selectedNode: null, selectedIteration: null });
});

function llmNode(iteration: number): LLMCallNode {
  return {
    type: "llm_call",
    sequence_index: iteration,
    iteration,
    input_tokens: 1,
    output_tokens: 1,
    has_tool_calls: false,
  } as unknown as LLMCallNode;
}

function withClient(client: QueryClient, ui: ReactNode) {
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

/** Load calls through the real client, then render the inspector for `iteration`. */
async function inspect(total: number, iteration: number, defaultView?: "semantic" | "provider") {
  const { requests, fetchMock } = fakeServer(total);
  vi.stubGlobal("fetch", fetchMock);
  const result: LLMCallsResult = await fetchLLMCalls("R");
  const client = new QueryClient();
  client.setQueryData(["llm-calls", "R"], result);
  useDashboardStore.setState({
    selectedNode: llmNode(iteration),
    selectedIteration: iteration,
  });
  const html = renderToStaticMarkup(
    withClient(client, <PayloadInspector runId="R" messages={{}} systemPrompt={null} defaultView={defaultView} />),
  );
  return { requests, result, html };
}

describe("LLM call paging", () => {
  it("loads call 201 with a second request and shows its stored payload", async () => {
    const { requests, result, html } = await inspect(201, 201);
    expect(requests).toEqual([0, 200]);
    expect(result.incomplete).toBe(false);
    expect(result.llm_calls).toHaveLength(201);
    expect(html).toContain("request-201");
    expect(html).not.toContain("more LLM calls than the dashboard loads");
    expect(html).not.toContain("Run the agent again");
  });

  it("loads exactly 1,000 calls in five requests and reports them complete", async () => {
    const { requests, result } = await inspect(1000, 1000);
    expect(requests).toEqual([0, 200, 400, 600, 800]);
    expect(result.incomplete).toBe(false);
  });

  it("stops at 1,000 and says the rest is stored but not loaded (call 1,001)", async () => {
    const { requests, result, html } = await inspect(1001, 1001);
    expect(requests).toHaveLength(5);
    expect(result.incomplete).toBe(true);
    expect(result.llm_calls).toHaveLength(1000);
    expect(html).toContain("more LLM calls than the dashboard loads");
    expect(html).toContain("it is stored");
    expect(html).not.toContain("Run the agent again");
    expect(html).not.toContain("request-1001");
  });
});

describe("provider view of the selected call", () => {
  it("call 1,001: stored but not loaded, with no advice to re-run", async () => {
    const { html } = await inspect(1001, 1001, "provider");
    expect(html).toContain("stored, but outside the range the dashboard loaded");
    expect(html).not.toContain("Run the agent again");
  });

  it("call 201 is loaded, so the provider view shows its payload section", async () => {
    const { html } = await inspect(201, 201, "provider");
    expect(html).toContain("Provider Request");
    expect(html).not.toContain("outside the range");
    expect(html).not.toContain("Run the agent again");
  });

  it("a call the run never recorded still gets the re-run advice", async () => {
    const { html } = await inspect(5, 99, "provider");
    expect(html).toContain("No provider payload data for this call");
    expect(html).toContain("Run the agent again");
  });
});

describe("provider empty state", () => {
  it("explains stored-but-unloaded data and never suggests re-running", () => {
    const html = renderToStaticMarkup(<ProviderEmptyState loaded={1000} />);
    expect(html).toContain("stored, but outside the range the dashboard loaded");
    expect(html).toContain("first 1000 LLM calls");
    expect(html).not.toContain("Run the agent again");
  });

  it("keeps the re-run advice only when the run truly has no payload", () => {
    const html = renderToStaticMarkup(<ProviderEmptyState loaded={null} />);
    expect(html).toContain("No provider payload data for this call");
    expect(html).toContain("Run the agent again");
  });
});

const summary = {
  run_id: "R",
  agent_name: "Agent",
  status: "success",
  model: "m",
  strategy: "react",
  input_text: "hi",
  answer: "ok",
  error: null,
  iteration_count: 1,
  total_input_tokens: 1,
  total_output_tokens: 1,
  total_cost_usd: null,
  total_cache_read_tokens: 0,
  total_cache_creation_tokens: 0,
  total_reasoning_tokens: 0,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function renderRunDetail(data: Partial<RunDetailResponse>) {
  const client = new QueryClient();
  client.setQueryData(["run-detail", "R"], {
    summary,
    nodes: [],
    system_prompt: null,
    messages_by_iteration: {},
    delegation: null,
    ...data,
  });
  return renderToStaticMarkup(
    withClient(
      client,
      <MemoryRouter initialEntries={["/runs/R"]}>
        <Routes>
          <Route path="/runs/:runId" element={<RunDetail />} />
        </Routes>
      </MemoryRouter>,
    ),
  );
}

describe("timeline and delegation warnings", () => {
  it("warns when the server flagged the timeline truncated", () => {
    const html = renderRunDetail({ truncated: true });
    expect(html).toContain("more events than the dashboard reads");
    expect(html).toContain('role="status"');
  });

  it("shows no warning for a complete timeline", () => {
    const html = renderRunDetail({ truncated: false });
    expect(html).not.toContain("more events than the dashboard reads");
  });

  const delegation = (truncated: boolean): DelegationInfo => ({
    parent: null,
    children: [{ run_id: "C1", agent_name: "Child", status: "success", delegation_level: 1 }],
    ancestry: [],
    subtree_summary: {
      direct_child_count: 1,
      descendant_count: 1,
      max_depth: 1,
      subtree_input_tokens: 1,
      subtree_output_tokens: 1,
      subtree_cost_usd: null,
      unknown_cost_count: 1,
      status_counts: { success: 1 },
    },
    ancestry_complete: true,
    truncated,
  });

  it("warns that a bounded delegation view is partial", () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <DelegationPanel delegation={delegation(true)} />
      </MemoryRouter>,
    );
    expect(html).toContain("delegation tree is very large");
    expect(html).toContain("lower bounds");
  });

  it("says nothing for a complete delegation view", () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <DelegationPanel delegation={delegation(false)} />
      </MemoryRouter>,
    );
    expect(html).not.toContain("delegation tree is very large");
  });
});
