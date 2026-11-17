"""Present-source filtering for the temporary split forbidden-import contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from importlinter.contracts.forbidden import ForbiddenContract
from importlinter.domain.contract import ContractCheck

if TYPE_CHECKING:
    from grimp import ImportGraph

NAME = "Split operation modules cannot import compatibility facades"
SOURCES = frozenset(
    [
        "dendrux.runtime._run_errors",
        "dendrux.runtime._run_pending",
        "dendrux.runtime._run_contracts",
        "dendrux.runtime._run_observers",
        "dendrux.runtime._run_events",
        "dendrux.runtime._run_tools",
        "dendrux.runtime._run_images",
        "dendrux.runtime._run_entry",
        "dendrux.runtime._run_start",
        "dendrux.runtime._run_outcome",
        "dendrux.runtime._run_execute",
        "dendrux.runtime._run_retry",
        "dendrux.runtime._run_stream",
        "dendrux.runtime._run_resume_prep",
        "dendrux.runtime._run_resume_replay",
        "dendrux.runtime._run_approval",
        "dendrux.runtime._run_resume_loop",
        "dendrux.runtime._run_resume",
        "dendrux.runtime._run_resume_stream",
        "dendrux.runtime.stores.sql._sql_host",
        "dendrux.runtime.stores.sql._sql_sessions",
        "dendrux.runtime.stores.sql._sql_support",
        "dendrux.runtime.stores.sql._sql_sweep_scan",
        "dendrux.runtime.stores.sql._sql_writes",
        "dendrux.runtime.stores.sql._sql_runs",
        "dendrux.runtime.stores.sql._sql_evidence",
        "dendrux.runtime.stores.sql._sql_managed",
        "dendrux.runtime.stores.sql._sql_tree",
        "dendrux.runtime.stores.sql._sql_lifecycle",
        "dendrux.runtime.stores.sql._sql_pause",
        "dendrux.runtime.stores.sql._sql_reads",
        "dendrux.runtime.stores.sql._sql_events",
        "dendrux.runtime.stores.sql._sql_claims",
        "dendrux.runtime.stores.sql._sql_leases",
        "dendrux.runtime.stores.sql._sql_sweep",
    ]
)


class PresentForbiddenContract(ForbiddenContract):
    """Apply the regular forbidden contract to operation modules that exist."""

    @classmethod
    def _get_field_names(cls) -> list[str]:
        return ForbiddenContract._get_field_names()

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        if self.name != NAME:
            return super().check(graph, verbose)
        sources: list[str] = self.contract_options["source_modules"]
        if set(sources) - SOURCES:
            raise ValueError("unknown split source module")
        options = dict(self.contract_options)
        options["source_modules"] = [name for name in sources if name in graph.modules]
        if not options["source_modules"]:
            return ContractCheck(kept=True, metadata={"invalid_chains": []})
        return ForbiddenContract(self.name, self.session_options, options).check(graph, verbose)
