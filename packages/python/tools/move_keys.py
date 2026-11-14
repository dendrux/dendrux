"""Prepare conserved quality-key transfers without stamping committed baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

GATES = {"size", "cognitive", "crap", "ruff", "vulture", "semgrep"}
RUFF_ACTION = (
    "conserve per rule by re-running Ruff on before and after sources; no per-function key"
)


def _destination(row: Mapping[str, Any]) -> str:
    if (
        set(row) != {"baseline", "new"}
        or set(row["new"]) != {"mixin"}
        or not isinstance(row["new"]["mixin"], str)
    ):
        raise ValueError("key map requires one exact lexical mixin destination")
    return str(row["new"]["mixin"])


def _kind(gate: str, old: str, new: str) -> None:
    if old == new:
        raise ValueError("key transfer must change identity")
    if gate == "size" and (":" not in old or ":" not in new):
        raise ValueError("new files receive no size allowance")
    if gate == "semgrep" and (len(old.split("|")) != 3 or old.split("|")[0] != new.split("|")[0]):
        raise ValueError("cannot move semgrep debt between rules")
    if gate == "vulture" and old.split(":")[-2:] != new.split(":")[-2:]:
        raise ValueError("cannot move vulture debt between names/kinds")


def _value(gate: str, old: Any, current: Any) -> Any:
    if gate == "crap":
        if not isinstance(current, dict) or set(current) != {
            "complexity",
            "coverage",
            "crap",
            "digest",
        }:
            raise ValueError("CRAP transfer needs fresh measured complexity/coverage/digest")
        if (
            any(
                not math.isfinite(current[key]) or current[key] < 0
                for key in ("complexity", "crap", "coverage")
            )
            or current["coverage"] > 1
            or not isinstance(current["digest"], str)
            or not current["digest"]
        ):
            raise ValueError("invalid fresh CRAP measurement")
        if any(current[key] > old[key] for key in ("complexity", "crap")):
            raise ValueError("CRAP/complexity allowance grew")
        return dict(current)
    if not isinstance(current, int) or isinstance(current, bool) or current < 0 or current > old:
        raise ValueError("allowance grew or invalid measurement")
    threshold = {"size": 80, "cognitive": 15}.get(gate, 0)
    return current if current > threshold else None


def _ruff(
    baseline: Mapping[str, Any],
    scores: Mapping[str, Any],
    mapping: Mapping[str, Any],
    present: Collection[str],
) -> dict[str, Any]:
    modules: dict[str, set[str]] = {}
    for gate in ("size", "cognitive"):
        for old, row in mapping.get(gate, {}).items():
            new = _destination(row)
            modules.setdefault(old.split(":")[0], set()).add(new.split(":")[0])
    for old, row in mapping.get("crap", {}).items():
        new = _destination(row)
        source = old.split("::")[0].removeprefix("src/").removesuffix(".py").replace("/", ".")
        target = new.split("::")[0].removeprefix("src/").removesuffix(".py").replace("/", ".")
        modules.setdefault(source, set()).add(target)
    updated = dict(baseline)
    destinations: set[str] = set()
    for old, row in mapping.get("ruff", {}).items():
        if (
            set(row) != {"baseline", "action"}
            or row["action"] != RUFF_ACTION
            or baseline.get(old) != row["baseline"]
        ):
            raise ValueError("ruff source/action/baseline mismatch")
        source, rule = old.rsplit(":", 1)
        keys = {
            f"{module}:{rule}"
            for module in modules.get(source, set())
            if module in present or f"{module}:{rule}" in present
        }
        keys.add(old)
        claims = {key: scores.get(key, 0) for key in keys}
        if (
            any(not isinstance(v, int) or v < 0 for v in claims.values())
            or sum(claims.values()) > baseline[old]
        ):
            raise ValueError("ruff retained plus moved allowance grew")
        for key, value in claims.items():
            if key != old and value and (key in baseline or key in destinations):
                raise ValueError("duplicate ruff allowance")
            if value:
                updated[key] = value
                destinations.add(key)
            else:
                updated.pop(key, None)
    return updated


def apply_key_map(
    gate: str,
    baseline: Mapping[str, Any],
    scores: Mapping[str, Any],
    mapping: Mapping[str, Any],
    present: Collection[str],
) -> dict[str, Any]:
    """Return a proposed baseline; every transferred cap is conserved or shrunk.

    The full v4 map may include future moves: a destination absent while its source
    remains is inactive. Missing both identities refuses the transfer.
    """
    if gate not in GATES or gate not in mapping:
        raise ValueError("unknown/missing quality gate")
    if gate == "ruff":
        return _ruff(baseline, scores, mapping, present)
    updated = dict(baseline)
    destinations: set[str] = set()
    for old, row in mapping[gate].items():
        new = _destination(row)
        _kind(gate, old, new)
        if new not in present and old in present:
            continue
        if old not in baseline or baseline[old] != row["baseline"]:
            raise ValueError("source baseline differs from reviewed map")
        if old in present or new not in present or new not in scores:
            raise ValueError("source remains or destination measurement missing")
        if new in baseline or new in destinations:
            raise ValueError("destination allowance already claimed")
        destinations.add(new)
        current = _value(gate, baseline[old], scores[new])
        del updated[old]
        if current is not None:
            updated[new] = current
    return updated


def exemption_transfers(mapping: Mapping[str, Any], exemptions: Collection[str]) -> dict[str, str]:
    """Resolve semgrep allow_in identities through the real lexical CRAP owner map."""
    owners = {}
    for old, row in mapping.get("crap", {}).items():
        path, qual = old.split("::", 1)
        new_path, new_qual = _destination(row).split("::", 1)
        module = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        new_module = new_path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        owners[f"{module}:{qual}"] = f"{new_module}:{new_qual}"
    missing = set(exemptions) - set(owners)
    if missing:
        raise ValueError("semgrep exemption has no exact reviewed owner")
    return {key: owners[key] for key in exemptions}


def main() -> int:
    """Print a proposal only; no baseline file is written by this command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=sorted(GATES))
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--present", type=Path, required=True)
    args = parser.parse_args()
    try:
        proposal = apply_key_map(
            args.gate,
            json.loads(args.baseline.read_text()),
            json.loads(args.scores.read_text()),
            json.loads(args.map.read_text()),
            json.loads(args.present.read_text()),
        )
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"move_keys: refused: {exc}\n")
    print(json.dumps(proposal, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
