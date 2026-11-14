"""Validate reviewed move evidence and normalize only enumerated syntax edits."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import re
import tokenize
from typing import TYPE_CHECKING, Any

from tools._scopes import ScopeIndex, dotted, resolve_dotted, resolve_package
from tools.size_ratchet import function_sizes

if TYPE_CHECKING:
    from collections.abc import Mapping

FIELDS = {
    "version",
    "base",
    "old",
    "source_sha256",
    "sources",
    "owners",
    "lexical",
    "hoists",
    "annotations",
    "metadata",
    "scaffolding",
    "effects",
    "seams",
    "providers",
}
EFFECTS = {"logger", "_STOP_REASONS", "_UNSET_DEPTH", "_WAIT_MARK_LABELS"}
ANNOTATIONS = {
    "_CheckedSessionFactory": "_StoreHost",
    "_creation_mark_proven": "_StoreHost",
    "_creation_metadata": "_StoreHost",
    "_SweepScan": "_SweepStore",
}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
SAFE_STDLIB = frozenset(
    {
        "math",
        "itertools",
        "types",
        "decimal",
        "contextlib",
        "dataclasses",
        "datetime",
        "functools",
        "inspect",
        "json",
        "logging",
        "sys",
        "asyncio",
    }
)


def digest(source: str) -> str:
    """Hash exact UTF-8 source, including formatting and comments."""
    return hashlib.sha256(source.encode()).hexdigest()


def evidence(manifest: Mapping[str, Any], old: str, after: Mapping[str, str], old_key: str) -> None:
    """Refuse incomplete, stale or unrecognized evidence before any normalization."""
    if set(manifest) != FIELDS or manifest["version"] != 1:
        raise ValueError("manifest schema/version mismatch")
    if not re.fullmatch("[0-9a-f]{40}", manifest["base"]):
        raise ValueError("manifest requires immutable 40-character base SHA")
    if manifest["old"] != old_key or manifest["source_sha256"] != digest(old):
        raise ValueError("manifest original source hash/path mismatch")
    if manifest["sources"] != {k: digest(v) for k, v in after.items()}:
        raise ValueError("manifest cumulative destination source hashes mismatch")
    if set(manifest["effects"]) - EFFECTS or len(set(manifest["effects"])) != len(
        manifest["effects"]
    ):
        raise ValueError("manifest only permits named singleton effects once")
    for field in ("lexical", "hoists", "annotations", "metadata"):
        if not isinstance(manifest[field], list):
            raise ValueError(f"manifest {field} must be a list")
    for field in ("owners", "sources", "seams", "scaffolding", "providers"):
        if not isinstance(manifest[field], dict):
            raise ValueError(f"manifest {field} must be an object")


def roots(tree: ast.Module, name: str) -> list[ast.stmt]:
    """Find a top-level root or its uniquely relocated class member."""
    parent, sep, member = name.partition(".")
    if not sep:
        return [
            n
            for n in tree.body
            if getattr(n, "name", None) == name
            or (
                isinstance(n, ast.Assign | ast.AnnAssign)
                and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in (n.targets if isinstance(n, ast.Assign) else [n.target])
                )
            )
        ]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    original = [n for n in classes if n.name == parent]
    return [
        n for cls in original or classes for n in cls.body if getattr(n, "name", None) == member
    ]


def entry_fields(entry: Mapping[str, Any], fields: set[str]) -> None:
    if set(entry) != fields:
        raise ValueError("manifest entry has missing or unknown fields")


def direct_import(index: ScopeIndex, name: str, target: str) -> bool:
    binding = index.module.resolve(name)
    return bool(binding and binding.kind == "import" and binding.direct and binding.value == target)


def _old_global(index: ScopeIndex, node: ast.Name, old_module: str, name: str) -> str | None:
    scope = index.scope_of(node)
    if scope.find(name) is not index.module:
        return None
    binding = scope.resolve(name)
    if binding is None:
        return None
    if binding.kind == "import":
        return binding.value if binding.direct else None
    return f"{old_module}.{name}"


def lexical(
    old_tree: ast.Module,
    trees: Mapping[str, ast.Module],
    entry: Mapping[str, Any],
    old_module: str,
    modules: Mapping[str, str],
    owners: Mapping[str, str],
    seams: Mapping[str, str],
) -> None:
    entry_fields(entry, {"root", "module", "name", "target", "count"})
    root, module, name, target = (entry[k] for k in ("root", "module", "name", "target"))
    if module not in trees:
        raise ValueError(f"lexical module unknown: {module} (root {root})")
    if entry["count"] < 1:
        raise ValueError("lexical count must be positive")
    old_roots, new_roots = roots(old_tree, root), roots(trees[module], root)
    if len(old_roots) != len(new_roots) or not old_roots:
        raise ValueError(f"lexical root unknown or ambiguous: {root} in {module}")
    before = ScopeIndex(old_tree, old_module.rpartition(".")[0])
    now = ScopeIndex(trees[module], modules[module].rpartition(".")[0])
    if name not in seams and (name not in owners or owners[name] not in modules):
        raise ValueError(f"lexical binding owner unknown: {name} (root {root})")
    expected = seams.get(name) or f"{modules[owners[name]]}.{name}"
    if target != expected:
        raise ValueError("lexical target is not the declared owner")
    old_reads = [
        n
        for root_node in old_roots
        for n in ast.walk(root_node)
        if isinstance(n, ast.Name)
        and isinstance(n.ctx, ast.Load)
        and n.id == name
        and _old_global(before, n, old_module, name)
    ]
    if len(old_reads) != entry["count"]:
        raise ValueError("lexical original count mismatch (shadowed/rebound/type-only name)")
    for read in old_reads:
        actual = _old_global(before, read, old_module, name)
        if actual != (
            "dendrux.skills._session.current"
            if name == "current_skill_session"
            and target == "dendrux.runtime._run_entry.current_skill_session"
            else seams.get(name) or f"{old_module}.{name}"
        ):
            raise ValueError("lexical original binding mismatch")
    matches = []
    for root_node in new_roots:
        for node in ast.walk(root_node):
            if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
                continue
            spelling = dotted(node)
            if spelling is None or resolve_dotted(spelling, now.scope_of(node)) != target:
                continue
            head = spelling.split(".")[0]
            binding = now.scope_of(node).resolve(head)
            if now.scope_of(node).find(head) is not now.module or not binding or not binding.direct:
                raise ValueError("lexical owner is shadowed or not unconditionally imported")
            if not now.scope_of(node).deferred() and binding.line >= node.lineno:
                raise ValueError("lexical owner is imported after eager default/decorator use")
            matches.append(node)
    if len(matches) != entry["count"]:
        raise ValueError("lexical replacement count/binding mismatch")
    replacements = {id(n) for n in matches}

    class Restore(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            if id(node) in replacements:
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            return self.generic_visit(node)

    Restore().visit(trees[module])


def _comments(source: str, first: int, last: int) -> str:
    return "\n".join(
        t.string
        for t in tokenize.generate_tokens(io.StringIO(source).readline)
        if t.type == tokenize.COMMENT and first <= t.start[0] <= last
    )


def hoist(
    old_tree: ast.Module,
    trees: Mapping[str, ast.Module],
    entry: Mapping[str, Any],
    old: str,
    after: Mapping[str, str],
    old_module: str,
) -> None:
    entry_fields(
        entry,
        {"root", "module", "statement", "comment"} | ({"eager"} if "eager" in entry else set()),
    )
    candidates = roots(old_tree, entry["root"])
    statement = ast.parse(entry["statement"]).body
    if (
        len(candidates) != 1
        or len(statement) != 1
        or not isinstance(statement[0], ast.Import | ast.ImportFrom)
    ):
        raise ValueError("hoist root/import mismatch")
    node = candidates[0]
    if not isinstance(node, FUNCTIONS):
        raise ValueError("hoist requires a function root")
    leading = []
    for child in node.body:
        if (
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Constant)
            and isinstance(child.value.value, str)
        ):
            continue
        if not isinstance(child, ast.Import | ast.ImportFrom):
            break
        leading.append(child)
    wanted = ast.dump(statement[0])
    found = [n for n in leading if ast.dump(n) == wanted]
    destination = [n for n in trees[entry["module"]].body if ast.dump(n) == wanted]
    if len(found) != 1 or len(destination) != 1:
        raise ValueError("only unconditional leading imports may hoist once")
    before, now = found[0], destination[0]
    import_root = (
        before.names[0].name.split(".")[0]
        if isinstance(before, ast.Import)
        else (before.module or "").split(".")[0]
    )
    _safe_hoist(before, old_tree, old_module, entry.get("eager", []), import_root)
    old_index = ScopeIndex(old_tree)
    new_index = ScopeIndex(trees[entry["module"]])
    for alias in before.names:
        local = alias.asname or (
            alias.name.split(".")[0] if isinstance(before, ast.Import) else alias.name
        )
        reads = [
            (n, s)
            for n, s in old_index.names
            if n.id == local
            and n.lineno >= node.lineno
            and n.lineno <= (node.end_lineno or node.lineno)
        ]
        bindings = [s.resolve(local) for _, s in reads]
        if any(binding is None or binding.kind != "import" for binding in bindings):
            raise ValueError("hoist changes a shadowed/rebound/parameter binding")
        if new_index.module.resolve(local) is None:
            raise ValueError("hoisted import has ambiguous module binding")
    # A module import must precede every executable definition/default that can read it.
    preceding = trees[entry["module"]].body[: trees[entry["module"]].body.index(now)]
    if any(
        not isinstance(n, ast.Import | ast.ImportFrom)
        and not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        )
        for n in preceding
    ):
        raise ValueError("hoisted import follows executable statements")
    for text, item in ((old, before), (after[entry["module"]], now)):
        if _comments(text, item.lineno, item.end_lineno or item.lineno) != entry["comment"]:
            raise ValueError("hoist comments changed")
    node.body.remove(before)


def normalize_annotations(
    old_tree: ast.Module, trees: Mapping[str, ast.Module], entry: Mapping[str, Any]
) -> None:
    entry_fields(entry, {"root", "module", "host", "count"})
    root, host = entry["root"], entry["host"]
    if ANNOTATIONS.get(root) != host:
        raise ValueError("unreviewed store annotation change")
    count = 0
    for node in roots(trees[entry["module"]], root):
        for child in ast.walk(node):
            value = (
                child.annotation
                if isinstance(child, ast.arg | ast.AnnAssign)
                else child.returns
                if isinstance(child, FUNCTIONS)
                else None
            )
            if value is not None:
                for ref in ast.walk(value):
                    if isinstance(ref, ast.Name) and ref.id == host:
                        ref.id = "SQLStateStore"
                        count += 1
                    elif isinstance(ref, ast.Constant) and ref.value == host:
                        ref.value = "SQLStateStore"
                        count += 1
    if count != entry["count"] or count < 1 or not roots(old_tree, root):
        raise ValueError("store annotation entry unused/mismatched")


def metadata(
    old_tree: ast.Module, trees: Mapping[str, ast.Module], entry: Mapping[str, Any], old_module: str
) -> None:
    entry_fields(entry, {"root", "module", "attribute", "value"})
    root, attr, value = (entry[k] for k in ("root", "attribute", "value"))
    original = roots(old_tree, root)
    if not original or "." in root or attr not in {"__module__", "__qualname__"}:
        raise ValueError("metadata target is not an original class/public runner function")
    cls = isinstance(original[0], ast.ClassDef)
    public_runner = (
        old_module == "dendrux.runtime.runner"
        and not root.startswith("_")
        and isinstance(original[0], FUNCTIONS)
    )
    if not cls and not public_runner:
        raise ValueError("SQL method/private function metadata is not permitted")
    if value != (old_module if attr == "__module__" else root):
        raise ValueError("metadata value differs from original identity")
    wanted = ast.dump(ast.parse(f"{root}.{attr} = {value!r}").body[0])
    tree = trees[entry["module"]]
    found = [n for n in tree.body if ast.dump(n) == wanted]
    if len(found) != 1:
        raise ValueError("metadata pin missing/duplicated/mismatched")
    definitions = roots(tree, root)
    if not definitions or tree.body.index(found[0]) <= max(tree.body.index(n) for n in definitions):
        raise ValueError(f"metadata pin must follow its definition: {root}")
    tree.body.remove(found[0])


def scaffolding(
    old_tree: ast.Module, trees: Mapping[str, ast.Module], entries: Mapping[str, Any]
) -> None:
    for name, entry in entries.items():
        if name not in {"_StoreHost", "_SweepStore"}:
            raise ValueError("only reviewed host contracts may be new")
        entry_fields(entry, {"module", "source"})
        expected = ast.parse(entry["source"]).body
        found = roots(trees[entry["module"]], name)
        if len(found) != 1 or len(expected) != 1 or ast.dump(found[0]) != ast.dump(expected[0]):
            raise ValueError("contract scaffold source mismatch")
        cls = found[0]
        if not isinstance(cls, ast.ClassDef) or cls.decorator_list or cls.keywords:
            raise ValueError("invalid contract scaffold")
        guards = ScopeIndex(trees[entry["module"]])
        signatures = {n.name: n for n in ast.walk(old_tree) if isinstance(n, FUNCTIONS)}
        if name == "_StoreHost" and cls.bases:
            raise ValueError("_StoreHost must be annotation-only, without runtime bases")
        if name == "_SweepStore" and (
            len(cls.bases) != 1
            or resolve_dotted(dotted(cls.bases[0]) or "", guards.module) != "typing.Protocol"
        ):
            raise ValueError("_SweepStore must be the reviewed structural Protocol")
        for child in cls.body:
            if (
                isinstance(child, ast.Expr)
                and isinstance(child.value, ast.Constant)
                and isinstance(child.value.value, str)
            ):
                continue
            if isinstance(child, ast.AnnAssign) and child.value is None:
                if not isinstance(child.target, ast.Name) or child.target.id in signatures:
                    raise ValueError("method Any/object slots are not provider inheritance")
                continue
            declarations = (
                child.body
                if isinstance(child, ast.If)
                and not child.orelse
                and resolve_dotted(dotted(child.test) or "", guards.module)
                == "typing.TYPE_CHECKING"
                else [child]
                if name == "_SweepStore"
                else []
            )
            if not declarations:
                raise ValueError("_StoreHost must have no runtime members")
            for declaration in declarations:
                if not isinstance(declaration, FUNCTIONS) or declaration.name not in signatures:
                    raise ValueError("scaffold duplicates implementation or unknown signature")
                original = signatures[declaration.name]
                if (
                    type(declaration) is not type(original)
                    or ast.dump(declaration.args) != ast.dump(original.args)
                    or ast.dump(declaration.returns or ast.Constant(None))
                    != ast.dump(original.returns or ast.Constant(None))
                ):
                    raise ValueError("scaffold erased/changed retained signature")
                if (
                    len(declaration.body) != 1
                    or not isinstance(declaration.body[0], ast.Expr)
                    or not isinstance(declaration.body[0].value, ast.Constant)
                    or declaration.body[0].value.value is not Ellipsis
                ):
                    raise ValueError("scaffold contains executable implementation")


def normalize(
    old: str,
    after: Mapping[str, str],
    manifest: Mapping[str, Any],
    old_key: str,
    modules: Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    """Return comparison copies; leave original source available for hash/span checks."""
    evidence(manifest, old, after, old_key)
    old_tree = ast.parse(old)
    trees = {key: ast.parse(source) for key, source in after.items()}
    scaffolding(old_tree, trees, manifest["scaffolding"])
    original = copy.deepcopy(old_tree)
    _validate_seams(original, manifest)
    _pin_logger(old_tree, manifest, old_key, modules[old_key])
    for node in old_tree.body:
        name = getattr(node, "name", None)
        owner = manifest["owners"].get(name)
        if (
            isinstance(name, str)
            and owner
            and owner != old_key
            and (
                isinstance(node, ast.ClassDef)
                or (
                    isinstance(node, FUNCTIONS)
                    and modules[old_key] == "dendrux.runtime.runner"
                    and not name.startswith("_")
                )
            )
        ):
            pins = [
                e
                for e in manifest["metadata"]
                if e.get("root") == name
                and e.get("attribute") == "__module__"
                and e.get("module") == owner
            ]
            if len(pins) != 1:
                raise ValueError(f"missing exact original metadata pin for {name}")
    for entry in manifest["hoists"]:
        hoist(old_tree, trees, entry, old, after, modules[old_key])
    for entry in manifest["lexical"]:
        lexical(
            original, trees, entry, modules[old_key], modules, manifest["owners"], manifest["seams"]
        )
    for entry in manifest["annotations"]:
        normalize_annotations(original, trees, entry)
    for entry in manifest["metadata"]:
        metadata(original, trees, entry, modules[old_key])
    return ast.unparse(old_tree), {key: ast.unparse(tree) for key, tree in trees.items()}


def _validate_seams(old: ast.Module, manifest: Mapping[str, Any]) -> None:
    index = ScopeIndex(old)
    for name, target in manifest["seams"].items():
        binding = index.module.resolve(name)
        consumed = any(
            e.get("name") == name and e.get("target") == target for e in manifest["lexical"]
        )
        if binding is None or binding.kind != "import" or not consumed:
            raise ValueError(
                f"unused seam: {name}; require an original import and lexical consumer"
            )


def check_spans(old: str, after: Mapping[str, str], owners: Mapping[str, str]) -> list[str]:
    """Measure original physical spans, including the residual facade and nested functions."""
    before = function_sizes(old)
    sizes = {key: function_sizes(source) for key, source in after.items()}
    problems = []
    for qual, length in before.items():
        if length <= 80:
            continue
        root = next(
            (
                key
                for key in sorted(owners, key=len, reverse=True)
                if qual == key or qual.startswith(key + ".")
            ),
            None,
        )
        if root is None:
            problems.append(f"span: missing owner for {qual}")
            continue
        module = owners[root]
        suffix = qual[len(root) :]
        candidates = [
            value
            for key, value in sizes[module].items()
            if key == root + suffix
            or ("." in root and key.endswith("." + root.split(".")[-1] + suffix))
        ]
        if len(candidates) != 1 or candidates[0] > length:
            problems.append(f"span: {qual} exceeds original {length} lines or is ambiguous")
    return problems


def verify_provider_sources(manifest: Mapping[str, Any], read_base: Any, read_now: Any) -> None:
    """Bind reviewed external class snapshots to immutable and unchanged source files."""
    for hoist_entry in manifest["hoists"]:
        for step in hoist_entry.get("eager", []):
            path = f"src/{step['module'].replace('.', '/')}.py"
            if (
                step["path"] != path
                or digest(step["source"]) != step["sha256"]
                or digest(read_base(path)) != step["sha256"]
                or digest(read_now(path)) != step["sha256"]
            ):
                raise ValueError("eager dependency source hash/path mismatch")
    for identity, entry in manifest["providers"].items():
        module, _, name = identity.rpartition(".")
        path = f"src/{module.replace('.', '/')}.py"
        if entry["path"] != path:
            raise ValueError("provider source path differs from imported identity")
        source = read_base(path)
        if digest(source) != entry["sha256"] or digest(read_now(path)) != entry["sha256"]:
            raise ValueError("provider source hash changed from immutable base")
        definitions = [
            n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name
        ]
        snapshot = ast.parse(entry["class"]).body
        if (
            len(definitions) != 1
            or len(snapshot) != 1
            or ast.dump(definitions[0]) != ast.dump(snapshot[0])
        ):
            raise ValueError("provider class snapshot differs from immutable source")


def _imports(tree: ast.Module, module: str) -> set[str]:
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            base = resolve_package(module.rpartition(".")[0], node.level, node.module)
            if base:
                imported.add(base)
                imported.update(f"{base}.{a.name}" for a in node.names)
    return imported


def _safe_hoist(
    statement: ast.Import | ast.ImportFrom,
    old: ast.Module,
    old_module: str,
    eager: list[dict[str, str]],
    import_root: str,
) -> None:
    if isinstance(statement, ast.ImportFrom) and statement.level:
        raise ValueError("hoists require an exact absolute binding source")
    if isinstance(statement, ast.Import) and len(statement.names) != 1:
        raise ValueError("hoist imports must identify one binding source")
    target = statement.module if isinstance(statement, ast.ImportFrom) else statement.names[0].name
    if target in SAFE_STDLIB:
        if eager:
            raise ValueError("unused eager import evidence")
        return
    imported = _imports(old, old_module)
    for step in eager:
        entry_fields(step, {"module", "path", "sha256", "source"})
        if step["module"] not in imported:
            raise ValueError("eager import evidence is not an unconditional original dependency")
        imported = _imports(ast.parse(step["source"]), step["module"])
    if target not in imported:
        raise ValueError("optional/lazy import boundary cannot be hoisted")


def _pin_logger(
    old: ast.Module, manifest: Mapping[str, Any], old_key: str, old_module: str
) -> None:
    if "logger" not in manifest["effects"] or manifest["owners"].get("logger") == old_key:
        return
    found = roots(old, "logger")
    if len(found) != 1 or not isinstance(found[0], ast.Assign):
        raise ValueError("logger must have exactly one original assignment")
    call = found[0].value
    if not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
        raise ValueError("logger must retain the original getLogger call")
    index = ScopeIndex(old)
    if resolve_dotted(dotted(call.func) or "", index.module) != "logging.getLogger":
        raise ValueError("logger constructor binding differs from logging.getLogger")
    if isinstance(call.args[0], ast.Name) and call.args[0].id == "__name__":
        call.args[0] = ast.Constant(old_module)
    elif not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
        raise ValueError("logger requires the original literal module name")
