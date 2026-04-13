#!/usr/bin/env python3
"""
Generate markdown documentation for Fast-LLM configuration classes.

Walks the fast_llm package, finds all @config_class-decorated classes, and writes
one markdown file per class under docs/reference/configuration/, mirroring the
package structure. Also writes index.md files per directory and updates the nav
section in mkdocs.yaml.

Usage:
    python tools/generate_config_docs.py
"""

import dataclasses
import importlib
import pathlib
import pkgutil
import re
import sys
import types
import typing

from fast_llm.config import Config, Field, FieldHint, FieldHintImportance  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "reference" / "configuration"
MKDOCS_YAML = REPO_ROOT / "mkdocs.yaml"

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Field filtering
# ---------------------------------------------------------------------------

# Hints that describe internal/computed/testing fields — not useful in config docs.
EXCLUDED_HINTS: set[FieldHint] = {FieldHint.derived, FieldHint.setup, FieldHint.testing}

# Field names that are always excluded regardless of hint.
EXCLUDED_FIELD_NAMES: set[str] = {"type"}


def is_user_field(name: str, field: Field) -> bool:
    """Return True if this field should appear in user-facing documentation."""
    if name.startswith("_"):
        return False
    if name in EXCLUDED_FIELD_NAMES:
        return False
    if not field.init or field._field_type is not dataclasses._FIELD:  # noqa: SLF001
        return False
    if getattr(field, "hint", None) in EXCLUDED_HINTS:
        return False
    return True


# ---------------------------------------------------------------------------
# Module collection
# ---------------------------------------------------------------------------


def import_all_config_modules() -> None:
    """Import every module in the fast_llm package so all Config subclasses are registered."""
    import fast_llm  # noqa: F401

    for module_info in pkgutil.walk_packages(
        path=[str(REPO_ROOT / "fast_llm")],
        prefix="fast_llm.",
        onerror=lambda name: None,
    ):
        # Only import config modules — they are safe to import without GPU.
        if not module_info.name.endswith(".config"):
            continue
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {module_info.name}: {exc}", file=sys.stderr)


def collect_config_classes() -> dict[type, dict]:
    """
    Return a dict mapping each Config subclass to metadata:
        {
            "module": str,
            "fields": list[(name, Field, resolved_type)],
            "registry": dict[str, type] | None,  # subclasses if this has a registry
            "registered_in": list[(base_cls, type_key)],  # registries this class is in
            "abstract": bool,
        }
    """
    import fast_llm.config as config_module

    config_base = config_module.Config

    # Collect all Config subclasses that have been processed by @config_class.
    found: dict[type, dict] = {}
    for cls in _all_subclasses(config_base):
        if not getattr(cls, "__class_validated__", False):
            continue
        if cls.__module__ == "builtins":
            continue
        found[cls] = {
            "module": cls.__module__,
            "fields": [],
            "registry": None,
            "registered_in": [],
            "abstract": bool(getattr(cls, "_abstract", False)),
        }

    # Resolve type hints and build field lists.
    for cls, info in found.items():
        try:
            hints = typing.get_type_hints(cls)
        except Exception:  # noqa: BLE001
            hints = {}
        for name, field in cls.fields():
            if not is_user_field(name, field):
                continue
            resolved = hints.get(name, field.type)
            info["fields"].append((name, field, resolved))
        # Sort by hint importance (lower = more important), then alphabetically.
        info["fields"].sort(
            key=lambda t: (FieldHintImportance.get(getattr(t[1], "hint", FieldHint.unknown), 20), t[0])
        )

    # Build registry info.
    for cls, info in found.items():
        registry = getattr(cls, "_registry", None)
        if registry is not None:
            info["registry"] = {key: found_cls for key in registry if (found_cls := registry[key]) in found}

    # Build registered_in back-references.
    for cls, info in found.items():
        registry = getattr(cls, "_registry", None)
        if registry is None:
            continue
        for key in registry:
            subclass = registry[key]
            if subclass in found:
                found[subclass]["registered_in"].append((cls, key))

    return found


def _all_subclasses(cls: type) -> list[type]:
    """Recursively collect all subclasses of a class."""
    result = []
    queue = list(cls.__subclasses__())
    seen = set()
    while queue:
        sub = queue.pop()
        if sub in seen:
            continue
        seen.add(sub)
        result.append(sub)
        queue.extend(sub.__subclasses__())
    return result


# ---------------------------------------------------------------------------
# Back-reference computation
# ---------------------------------------------------------------------------


def build_back_refs(found: dict[type, dict]) -> dict[type, list[tuple[type, str]]]:
    """
    For each config class, find all (owner_class, field_name) pairs that reference it
    as part of their field type.
    """
    back_refs: dict[type, list[tuple[type, str]]] = {cls: [] for cls in found}

    for owner_cls, info in found.items():
        for name, _field, resolved_type in info["fields"]:
            for referenced_cls in _extract_config_types(resolved_type, found):
                back_refs[referenced_cls].append((owner_cls, name))

    return back_refs


def _extract_config_types(annotation, found: dict[type, dict]) -> list[type]:
    """Extract all Config subclass types referenced in an annotation."""
    results = []
    if isinstance(annotation, type) and annotation in found:
        results.append(annotation)
    elif isinstance(annotation, types.UnionType) or (
        hasattr(annotation, "__origin__") and annotation.__origin__ is typing.Union
    ):
        for arg in typing.get_args(annotation):
            results.extend(_extract_config_types(arg, found))
    elif hasattr(annotation, "__origin__"):
        for arg in typing.get_args(annotation):
            results.extend(_extract_config_types(arg, found))
    return results


# ---------------------------------------------------------------------------
# Type rendering
# ---------------------------------------------------------------------------


def render_type(
    annotation,
    found: dict[type, dict],
    cls_output_paths: dict[type, pathlib.Path],
    own_path: pathlib.Path,
) -> str:
    """Render a type annotation as a markdown string, linking to Config class pages."""
    if annotation is type(None):
        return "`None`"
    if annotation is typing.Any:
        return "`Any`"
    if isinstance(annotation, type):
        if annotation in found:
            rel_path = cls_output_paths.get(annotation)
            if rel_path is not None:
                link = _relative_link(own_path, rel_path)
                return f"[{annotation.__name__}]({link})"
            return f"`{annotation.__name__}`"
        if issubclass(annotation, type):
            return "`type`"
        return f"`{annotation.__name__}`"
    if isinstance(annotation, types.UnionType) or (
        hasattr(annotation, "__origin__") and annotation.__origin__ is typing.Union
    ):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        none_part = " or `None`" if type(None) in typing.get_args(annotation) else ""
        inner = " or ".join(render_type(a, found, cls_output_paths, own_path) for a in args)
        return inner + none_part
    if hasattr(annotation, "__origin__"):
        origin = annotation.__origin__
        args = typing.get_args(annotation)
        if origin is list:
            return f"list[{render_type(args[0], found, cls_output_paths, own_path)}]" if args else "`list`"
        if origin is dict:
            k = render_type(args[0], found, cls_output_paths, own_path) if args else "`Any`"
            v = render_type(args[1], found, cls_output_paths, own_path) if len(args) > 1 else "`Any`"
            return f"dict[{k}, {v}]"
        if origin is tuple:
            inner = ", ".join(render_type(a, found, cls_output_paths, own_path) for a in args)
            return f"tuple[{inner}]"
        if origin is set:
            return f"set[{render_type(args[0], found, cls_output_paths, own_path)}]" if args else "`set`"
        # Fallback for other generics
        return f"`{getattr(origin, '__name__', str(origin))}`"
    return f"`{annotation}`"


def render_default(field: Field, resolved_type, found: dict[type, dict]) -> str:
    """Render the default value of a field as a string."""
    if field.default is not dataclasses.MISSING:
        value = field.default
        if isinstance(value, str):
            return f'`"{value}"`'
        if value is None:
            return "`None`"
        # Class objects: show the class name, not `<class '...'>`
        if isinstance(value, type):
            return f"`{value.__name__}`"
        # Large integers: insert underscores every 3 digits for readability
        if isinstance(value, int) and abs(value) > 999_999:
            return f"`{value:_}`"
        return f"`{value}`"
    if field.default_factory is not dataclasses.MISSING:
        factory = field.default_factory
        # A factory that is itself a Config class means "instantiate with defaults".
        if isinstance(factory, type) and factory in found:
            return "*(sub-fields optional)*"
        if hasattr(factory, "__name__"):
            return f"`{factory.__name__}()`"
    # If the type itself is a Config class, the value is still required in YAML
    # but every sub-field within it has its own default — don't say "required".
    core_type = _unwrap_optional(resolved_type)
    if isinstance(core_type, type) and core_type in found:
        return "*(sub-fields optional)*"
    return "*(required)*"


def _unwrap_optional(annotation) -> type | None:
    """Return the inner type of Optional[X] / X | None, or the annotation itself."""
    if isinstance(annotation, types.UnionType) or (
        hasattr(annotation, "__origin__") and annotation.__origin__ is typing.Union
    ):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


# ---------------------------------------------------------------------------
# Output path computation
# ---------------------------------------------------------------------------


def get_module_dir(module_name: str) -> pathlib.Path:
    """
    Convert a module name like 'fast_llm.engine.distributed.config' to a
    relative output path like 'engine/distributed'.
    """
    parts = module_name.split(".")
    # Strip 'fast_llm' prefix.
    if parts and parts[0] == "fast_llm":
        parts = parts[1:]
    # Strip trailing 'config'.
    if parts and parts[-1] == "config":
        parts = parts[:-1]
    return pathlib.Path(*parts) if parts else pathlib.Path(".")


def compute_output_paths(found: dict[type, dict]) -> dict[type, pathlib.Path]:
    """
    Return a dict mapping each class to its output path relative to OUTPUT_DIR,
    e.g. engine/distributed/DistributedConfig.md
    """
    return {cls: get_module_dir(info["module"]) / f"{cls.__name__}.md" for cls, info in found.items()}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_hint_badge(hint: FieldHint) -> str:
    badge_map = {
        FieldHint.core: "core",
        FieldHint.architecture: "architecture",
        FieldHint.optional: "optional",
        FieldHint.performance: "performance",
        FieldHint.stability: "stability",
        FieldHint.feature: "feature",
        FieldHint.expert: "expert",
        FieldHint.logging: "logging",
        FieldHint.deprecated: "deprecated",
        FieldHint.wip: "wip",
        FieldHint.unknown: "",
    }
    label = badge_map.get(hint, str(hint))
    return f"`{label}`" if label else ""


def render_class_page(
    cls: type,
    info: dict,
    back_refs: list[tuple[type, str]],
    found: dict[type, dict],
    cls_output_paths: dict[type, pathlib.Path],
    own_path: pathlib.Path,
) -> str:
    """Render the full markdown page for a config class."""
    lines = []

    # Title
    lines.append(f"# {cls.__name__}\n")

    # Abstract badge
    if info["abstract"]:
        lines.append(
            '!!! note "Abstract"\n    This class cannot be instantiated directly. Use one of the variants listed below.\n'
        )

    # Module
    lines.append(f"**Module:** `{cls.__module__}`\n")

    # Registered as / variant of
    if info["registered_in"]:
        for base_cls, type_key in info["registered_in"]:
            base_path = cls_output_paths.get(base_cls)
            if base_path is not None:
                rel = _relative_link(own_path, base_path)
                lines.append(f"**Variant of:** [{base_cls.__name__}]({rel}) — select with `type: {type_key}`\n")
            else:
                lines.append(f"**Variant of:** `{base_cls.__name__}` — select with `type: {type_key}`\n")

    # Inheritance (Config parents only, skip Config itself and internal bases)
    config_parents = [
        base
        for base in cls.__mro__[1:]
        if base is not cls
        and isinstance(base, type)
        and issubclass(base, Config)
        and base.__name__ != "Config"
        and base in found
    ]
    if config_parents:
        parent_links = []
        for parent in config_parents[:3]:  # limit to 3 to avoid noise
            p_path = cls_output_paths.get(parent)
            if p_path is not None:
                rel = _relative_link(own_path, p_path)
                parent_links.append(f"[{parent.__name__}]({rel})")
            else:
                parent_links.append(f"`{parent.__name__}`")
        lines.append(f"**Inherits from:** {', '.join(parent_links)}\n")

    lines.append("")

    # Fields — definition list, one entry per field
    user_fields = info["fields"]
    if user_fields:
        lines.append("## Fields\n")
        for name, field, resolved_type in user_fields:
            type_str = render_type(resolved_type, found, cls_output_paths, own_path)
            default_str = render_default(field, resolved_type, found)
            hint = getattr(field, "hint", FieldHint.unknown)
            hint_str = render_hint_badge(hint)
            desc = getattr(field, "desc", None) or ""
            doc = getattr(field, "doc", None)
            if doc:
                desc = f"{desc} {doc}".strip() if desc else doc
            # Flatten multi-line descriptions (newlines break def-list indentation).
            desc = " ".join(desc.split())
            # Term: field name + hint badge (omit separator when hint is empty)
            term = f"`{name}`" + (f" — {hint_str}" if hint_str else "")
            lines.append(term)
            # Definition: metadata line, then description as a separate paragraph.
            meta = f"**Type:** {type_str} &nbsp;&nbsp; **Default:** {default_str}"
            lines.append(f":   {meta}")
            if desc:
                # Blank line + 4-space indent = new paragraph within the definition.
                lines.append(f"")
                lines.append(f"    {desc}")
            lines.append("")
    else:
        lines.append("*No user-configurable fields.*\n")

    # Variants table (if this class has a registry)
    registry = info.get("registry")
    if registry:
        lines.append("## Variants\n")
        lines.append("Select a variant by setting `type:` to one of the following values.\n")
        lines.append("| `type` value | Class | Description |")
        lines.append("|--------------|-------|-------------|")
        for key in sorted(registry):
            subclass = registry[key]
            sub_path = cls_output_paths.get(subclass)
            if sub_path is not None:
                rel = _relative_link(own_path, sub_path)
                class_link = f"[{subclass.__name__}]({rel})"
            else:
                class_link = f"`{subclass.__name__}`"
            sub_info = found.get(subclass, {})
            desc = _class_one_liner(subclass, sub_info)
            lines.append(f"| `{key}` | {class_link} | {desc} |")
        lines.append("")

    # Used in (back-references)
    if back_refs:
        lines.append("## Used in\n")
        seen = set()
        for owner_cls, field_name in sorted(back_refs, key=lambda t: (t[0].__name__, t[1])):
            key = (owner_cls, field_name)
            if key in seen:
                continue
            seen.add(key)
            owner_path = cls_output_paths.get(owner_cls)
            if owner_path is not None:
                rel = _relative_link(own_path, owner_path)
                lines.append(f"- [`{field_name}`]({rel}) in [{owner_cls.__name__}]({rel})")
            else:
                lines.append(f"- `{field_name}` in `{owner_cls.__name__}`")
        lines.append("")

    return "\n".join(lines)


def _class_one_liner(cls: type, info: dict) -> str:
    """Return a short description for a class, or empty string if none is available."""
    doc = getattr(cls, "__doc__", None)
    if doc:
        first_line = doc.strip().split("\n")[0].strip().rstrip(".")
        # Skip auto-generated __init__ signatures like "ClassName(**kwargs)"
        if first_line and not re.match(r"^\w.*\(.*\)\s*$", first_line):
            return first_line
    return ""


def _relative_link(from_path: pathlib.Path, to_path: pathlib.Path) -> str:
    """
    Compute a relative markdown link from one page to another,
    both paths relative to OUTPUT_DIR.
    """
    from_dir = from_path.parent
    try:
        rel = pathlib.Path(to_path).relative_to(from_dir)
    except ValueError:
        # Go up from from_dir to the common ancestor
        parts_from = from_dir.parts
        parts_to = to_path.parts
        # Find common prefix length
        common = 0
        for a, b in zip(parts_from, parts_to):
            if a == b:
                common += 1
            else:
                break
        up = len(parts_from) - common
        rel = pathlib.Path(*[".."] * up, *parts_to[common:])
    return str(rel).replace("\\", "/")


# ---------------------------------------------------------------------------
# Index page rendering
# ---------------------------------------------------------------------------


def render_index_page(
    directory: pathlib.Path,
    classes_in_dir: list[tuple[type, dict]],
    cls_output_paths: dict[type, pathlib.Path],
    subdirs: list[pathlib.Path],
) -> str:
    """Render an index.md for a directory."""
    lines = []

    # Title: use the directory name
    if directory == pathlib.Path("."):
        title = "Configuration Reference"
    else:
        title = " / ".join(p.replace("_", " ").title() for p in directory.parts)
    lines.append(f"# {title}\n")

    directory / "index.md"

    # Subdirectory links
    if subdirs:
        lines.append("## Sections\n")
        for subdir in sorted(subdirs):
            section_name = subdir.name.replace("_", " ").title()
            rel = str((subdir / "index.md").relative_to(directory)).replace("\\", "/")
            lines.append(f"- [{section_name}]({rel})")
        lines.append("")

    # Class table
    if classes_in_dir:
        lines.append("## Classes\n")
        lines.append("| Class | Description |")
        lines.append("|-------|-------------|")
        for cls, info in sorted(classes_in_dir, key=lambda t: t[0].__name__):
            cls_path = cls_output_paths[cls]
            rel = str(cls_path.relative_to(directory)).replace("\\", "/")
            desc = _class_one_liner(cls, info)
            abstract_note = " *(abstract)*" if info["abstract"] else ""
            lines.append(f"| [{cls.__name__}]({rel}){abstract_note} | {desc} |")
        lines.append("")

    return "\n".join(lines)


def render_root_index(
    found: dict[type, dict],
    cls_output_paths: dict[type, pathlib.Path],
    top_level_dirs: list[pathlib.Path],
) -> str:
    """Render the top-level index.md."""
    lines = [
        "# Configuration Reference\n",
        "This reference documents all configuration classes in Fast-LLM.",
        "Configurations are YAML files passed to the `fast-llm` CLI.",
        "The entry point is `GPTTrainerConfig`, which composes all other configurations.\n",
        "## Sections\n",
    ]
    for d in sorted(top_level_dirs):
        section_name = d.name.replace("_", " ").title()
        lines.append(f"- [{section_name}]({d.name}/index.md)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nav generation
# ---------------------------------------------------------------------------


def build_nav_tree(cls_output_paths: dict[type, pathlib.Path], found: dict[type, dict]) -> dict:
    """
    Build a nested dict representing the nav tree:
    { dir_path: { "index": index_path, "classes": [...], "subdirs": {subdir: ...} } }
    """
    tree: dict = {}

    for cls, rel_path in cls_output_paths.items():
        parts = rel_path.parent.parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        node.setdefault("_classes", []).append(cls)

    return tree


def nav_entries(
    tree: dict,
    cls_output_paths: dict[type, pathlib.Path],
    prefix: pathlib.Path = pathlib.Path("."),
) -> list:
    """Recursively build the mkdocs nav list for the config reference section."""
    entries = []

    # Index for this directory
    if prefix == pathlib.Path("."):
        index_rel = "reference/configuration/index.md"
    else:
        index_rel = f"reference/configuration/{prefix}/index.md".replace("\\", "/")
    entries.append(index_rel)

    # Classes directly in this directory
    classes = tree.get("_classes", [])
    for cls in sorted(classes, key=lambda c: c.__name__):
        rel = cls_output_paths[cls]
        entries.append(f"reference/configuration/{rel}".replace("\\", "/"))

    # Subdirectories
    for key, subtree in sorted((k, v) for k, v in tree.items() if not k.startswith("_")):
        subprefix = prefix / key if prefix != pathlib.Path(".") else pathlib.Path(key)
        section_name = key.replace("_", " ").title()
        sub_entries = nav_entries(subtree, cls_output_paths, subprefix)
        entries.append({section_name: sub_entries})

    return entries


def format_nav_yaml(entries: list, indent: int = 0) -> list[str]:
    """Render nav entries as YAML lines."""
    lines = []
    pad = "  " * indent
    for entry in entries:
        if isinstance(entry, str):
            lines.append(f"{pad}- {entry}")
        elif isinstance(entry, dict):
            for key, sub_entries in entry.items():
                lines.append(f"{pad}- {key}:")
                lines.extend(format_nav_yaml(sub_entries, indent + 1))
    return lines


# ---------------------------------------------------------------------------
# mkdocs.yaml nav update
# ---------------------------------------------------------------------------

NAV_SENTINEL_START = "  # BEGIN AUTO-GENERATED CONFIG REFERENCE"
NAV_SENTINEL_END = "  # END AUTO-GENERATED CONFIG REFERENCE"


def update_mkdocs_nav(nav_lines: list[str]) -> None:
    """
    Replace the auto-generated config reference section in mkdocs.yaml.
    If the sentinels are not present, append the section to the nav.
    """
    content = MKDOCS_YAML.read_text()

    new_block = "\n".join([NAV_SENTINEL_START] + nav_lines + [NAV_SENTINEL_END])

    if NAV_SENTINEL_START in content and NAV_SENTINEL_END in content:
        # Replace existing block
        pattern = re.escape(NAV_SENTINEL_START) + r".*?" + re.escape(NAV_SENTINEL_END)
        content = re.sub(pattern, new_block, content, flags=re.DOTALL)
    else:
        # Append before the last line of the nav section
        # Find the nav: key and append at the end of its list
        lines = content.splitlines()
        # Find the last non-empty line inside the nav block (heuristic: insert before next top-level key)
        insert_at = len(lines)
        in_nav = False
        for i, line in enumerate(lines):
            if line.startswith("nav:"):
                in_nav = True
            elif in_nav and line and not line.startswith(" "):
                insert_at = i
                break
        indent = "    "
        nav_indented = "\n".join(indent + l for l in new_block.splitlines())
        lines.insert(insert_at, nav_indented)
        content = "\n".join(lines) + "\n"

    MKDOCS_YAML.write_text(content)
    print(f"Updated nav in {MKDOCS_YAML}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(*, update_nav: bool = True, verbose: bool = True) -> None:
    """Generate all config reference docs, optionally updating mkdocs.yaml nav."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("Importing fast_llm config modules...")
    import_all_config_modules()

    log("Collecting config classes...")
    found = collect_config_classes()
    log(f"  Found {len(found)} config classes")

    log("Computing output paths...")
    cls_output_paths = compute_output_paths(found)

    log("Building back-references...")
    back_refs = build_back_refs(found)

    # Group classes by output directory
    dir_to_classes: dict[pathlib.Path, list[tuple[type, dict]]] = {}
    for cls, info in found.items():
        directory = cls_output_paths[cls].parent
        dir_to_classes.setdefault(directory, []).append((cls, info))

    log(f"Writing to {OUTPUT_DIR} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write class pages
    for cls, info in found.items():
        rel_path = cls_output_paths[cls]
        out_path = OUTPUT_DIR / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = render_class_page(cls, info, back_refs[cls], found, cls_output_paths, rel_path)
        out_path.write_text(content)

    # Write index pages — include all ancestor directories, not just leaf dirs with classes.
    leaf_dirs = {cls_output_paths[cls].parent for cls in found}
    all_dirs: set[pathlib.Path] = set()
    for directory in leaf_dirs:
        all_dirs.add(directory)
        for i in range(len(directory.parts)):
            all_dirs.add(pathlib.Path(*directory.parts[:i]) if i > 0 else pathlib.Path("."))

    # Find all top-level directories (direct children of output root)
    top_level_dirs = sorted({d.parts[0] for d in all_dirs if d != pathlib.Path(".")})

    for directory in sorted(all_dirs):
        classes_in_dir = dir_to_classes.get(directory, [])
        # Find immediate subdirectories
        subdirs = sorted(
            {
                directory / d.parts[len(directory.parts)]
                for d in all_dirs
                if len(d.parts) > len(directory.parts) and d.parts[: len(directory.parts)] == directory.parts
            }
        )
        index_content = render_index_page(directory, classes_in_dir, cls_output_paths, subdirs)
        index_path = OUTPUT_DIR / directory / "index.md"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(index_content)

    # Write root index
    root_index = render_root_index(
        found,
        cls_output_paths,
        [pathlib.Path(d) for d in top_level_dirs],
    )
    (OUTPUT_DIR / "index.md").write_text(root_index)

    if update_nav:
        log("Updating mkdocs.yaml nav...")
        tree = build_nav_tree(cls_output_paths, found)
        nav_root = nav_entries(tree, cls_output_paths)
        nav_yaml_lines = format_nav_yaml([{"Configuration Reference": nav_root}], indent=1)
        update_mkdocs_nav(nav_yaml_lines)

    log("Done.")


def main() -> None:
    generate(update_nav=True, verbose=True)


if __name__ == "__main__":
    main()
