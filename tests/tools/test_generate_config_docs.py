"""Unit tests for tools/generate_config_docs.py."""

import importlib.util
import pathlib
import typing

import pytest

from fast_llm.config import Config, Field, FieldHint, config_class

# ---------------------------------------------------------------------------
# Load the generator module via importlib (it is not a package).
# ---------------------------------------------------------------------------

_SCRIPT = pathlib.Path(__file__).parent.parent.parent / "tools" / "generate_config_docs.py"
_spec = importlib.util.spec_from_file_location("generate_config_docs", _SCRIPT)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


# ---------------------------------------------------------------------------
# Minimal synthetic Config classes used across multiple tests.
# ---------------------------------------------------------------------------


@config_class()
class _InnerConfig(Config):
    """A simple inner config for doc-generation tests."""

    value: int = Field(default=0, hint=FieldHint.core, desc="A value.")


@config_class()
class _OuterConfig(Config):
    """An outer config that references _InnerConfig."""

    inner: _InnerConfig = Field(hint=FieldHint.core, desc="Inner config.")
    required: str = Field(hint=FieldHint.core, desc="Required string field.")
    inner_optional: _InnerConfig | None = Field(default=None, hint=FieldHint.feature, desc="Optional inner.")
    string: str = Field(default="hello", hint=FieldHint.core, desc="A string.")
    large_int: int = Field(default=2**32, hint=FieldHint.core, desc="A large integer.")
    list_of_str: list[str] = Field(default_factory=list, hint=FieldHint.core, desc="A list of strings.")
    dict_field: dict[str, int] = Field(default_factory=dict, hint=FieldHint.core, desc="A dict.")


# Minimal `found` and `cls_output_paths` dicts used in render_* tests.
_FOUND: dict = {
    _InnerConfig: {
        "module": "tests.tools._InnerConfig",
        "fields": [],
        "registry": None,
        "registered_in": [],
        "abstract": False,
    },
    _OuterConfig: {
        "module": "tests.tools._OuterConfig",
        "fields": [],
        "registry": None,
        "registered_in": [],
        "abstract": False,
    },
}
_CLS_OUTPUT_PATHS: dict[type, pathlib.Path] = {
    _InnerConfig: pathlib.Path("tests/InnerConfig.md"),
    _OuterConfig: pathlib.Path("tests/OuterConfig.md"),
}
_OWN_PATH = pathlib.Path("tests/SomeConfig.md")

_OUTER_FIELDS = dict(_OuterConfig.fields())
_OUTER_HINTS = typing.get_type_hints(_OuterConfig)


# ---------------------------------------------------------------------------
# get_module_dir
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name, expected",
    [
        ("fast_llm.config", pathlib.Path(".")),
        ("fast_llm.engine.distributed.config", pathlib.Path("engine/distributed")),
        ("fast_llm.data.dataset.config", pathlib.Path("data/dataset")),
        ("fast_llm.models.gpt.config", pathlib.Path("models/gpt")),
        ("fast_llm.engine.training.config", pathlib.Path("engine/training")),
        # Module without trailing .config — just strip the fast_llm prefix.
        ("fast_llm.profile", pathlib.Path("profile")),
    ],
)
def test_get_module_dir(module_name, expected):
    assert _gen.get_module_dir(module_name) == expected


# ---------------------------------------------------------------------------
# _relative_link
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_path, to_path, expected",
    [
        # Same directory.
        ("engine/distributed/A.md", "engine/distributed/B.md", "B.md"),
        # Descend into child directory.
        ("engine/A.md", "engine/distributed/B.md", "distributed/B.md"),
        # Ascend to parent directory.
        ("engine/distributed/A.md", "engine/B.md", "../B.md"),
        # Sibling directory (up one, down one).
        ("engine/distributed/A.md", "engine/training/B.md", "../training/B.md"),
        # Deep cross-package link.
        ("engine/training/runner/A.md", "data/dataset/B.md", "../../../data/dataset/B.md"),
        # Top-level sibling packages.
        ("engine/A.md", "data/B.md", "../data/B.md"),
    ],
)
def test_relative_link(from_path, to_path, expected):
    assert _gen._relative_link(pathlib.Path(from_path), pathlib.Path(to_path)) == expected


# ---------------------------------------------------------------------------
# _unwrap_optional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (int | None, int),
        (str | None, str),
        (_InnerConfig | None, _InnerConfig),
        (int, int),
        (str, str),
    ],
)
def test_unwrap_optional_strips_none(annotation, expected):
    assert _gen._unwrap_optional(annotation) is expected


def test_unwrap_optional_union_unchanged():
    # Two non-None types: should not be simplified.
    annotation = int | str
    assert _gen._unwrap_optional(annotation) is annotation


def test_unwrap_optional_triple_union_unchanged():
    # Optional with two non-None types: should not be simplified.
    annotation = int | str | None
    assert _gen._unwrap_optional(annotation) is annotation


# ---------------------------------------------------------------------------
# render_hint_badge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hint, expected",
    [
        (FieldHint.core, "`core`"),
        (FieldHint.architecture, "`architecture`"),
        (FieldHint.optional, "`optional`"),
        (FieldHint.performance, "`performance`"),
        (FieldHint.feature, "`feature`"),
        (FieldHint.expert, "`expert`"),
        (FieldHint.logging, "`logging`"),
        (FieldHint.deprecated, "`deprecated`"),
        (FieldHint.wip, "`wip`"),
        # unknown → empty string (no badge).
        (FieldHint.unknown, ""),
    ],
)
def test_render_hint_badge(hint, expected):
    assert _gen.render_hint_badge(hint) == expected


# ---------------------------------------------------------------------------
# _class_one_liner
# ---------------------------------------------------------------------------


class _DocOneLiner:
    """A clean one-liner description."""


class _DocMultiLine:
    """First line only.

    More detail here that should be ignored.
    """


class _DocAutoSignature:
    """SomeName(**kwargs)"""


class _DocNoDocstring:
    pass


class _DocTrailingDot:
    """Description ending with a dot."""


@pytest.mark.parametrize(
    "cls, expected",
    [
        (_DocOneLiner, "A clean one-liner description"),
        (_DocMultiLine, "First line only"),
        (_DocAutoSignature, ""),  # auto-generated __init__ signature — filtered out
        (_DocNoDocstring, ""),
        (_DocTrailingDot, "Description ending with a dot"),  # trailing dot stripped
    ],
)
def test_class_one_liner(cls, expected):
    assert _gen._class_one_liner(cls, {}) == expected


# ---------------------------------------------------------------------------
# is_user_field — uses fields extracted from a synthetic Config class
# ---------------------------------------------------------------------------


@config_class()
class _IsUserFieldConfig(Config):
    normal: str = Field(default="x", hint=FieldHint.core, desc="Normal field.")
    feature: str = Field(default="x", hint=FieldHint.feature, desc="Feature field.")
    derived: str = Field(default="x", hint=FieldHint.derived, desc="Derived field.")
    testing: str = Field(default="x", hint=FieldHint.testing, desc="Testing field.")
    setup_field: str = Field(default="x", hint=FieldHint.setup, desc="Setup field.")


_IS_USER_FIELD_FIELDS = dict(_IsUserFieldConfig.fields())


@pytest.mark.parametrize(
    "field_name, expected",
    [
        ("normal", True),
        ("feature", True),
        ("derived", False),  # excluded hint
        ("testing", False),  # excluded hint
        ("setup_field", False),  # excluded hint
    ],
)
def test_is_user_field_hint(field_name, expected):
    assert _gen.is_user_field(field_name, _IS_USER_FIELD_FIELDS[field_name]) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("_private", False),  # underscore prefix → always excluded
        ("type", False),  # "type" is always excluded regardless of field content
        ("normal_name", True),
    ],
)
def test_is_user_field_name(name, expected):
    # Use a valid public field object; only the name varies.
    field = _IS_USER_FIELD_FIELDS["normal"]
    assert _gen.is_user_field(name, field) == expected


# ---------------------------------------------------------------------------
# _extract_config_types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation, expected_set",
    [
        (_InnerConfig, {_InnerConfig}),
        (_InnerConfig | None, {_InnerConfig}),
        (list[_InnerConfig], {_InnerConfig}),
        (dict[str, _InnerConfig], {_InnerConfig}),
        (_InnerConfig | _OuterConfig, {_InnerConfig, _OuterConfig}),
        (int, set()),
        (str | None, set()),
        (list[str], set()),
    ],
)
def test_extract_config_types(annotation, expected_set):
    result = _gen._extract_config_types(annotation, _FOUND)
    assert set(result) == expected_set


# ---------------------------------------------------------------------------
# render_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (str, "`str`"),
        (int, "`int`"),
        (bool, "`bool`"),
        (type(None), "`None`"),
        (typing.Any, "`Any`"),
        (str | None, "`str` or `None`"),
        (int | None, "`int` or `None`"),
        (list[str], "list[`str`]"),
        (list[int], "list[`int`]"),
        (dict[str, int], "dict[`str`, `int`]"),
        (tuple[str, int], "tuple[`str`, `int`]"),
        (set[str], "set[`str`]"),
    ],
)
def test_render_type_primitives(annotation, expected):
    assert _gen.render_type(annotation, _FOUND, _CLS_OUTPUT_PATHS, _OWN_PATH) == expected


def test_render_type_config_produces_link():
    result = _gen.render_type(_InnerConfig, _FOUND, _CLS_OUTPUT_PATHS, _OWN_PATH)
    # Should be a markdown link to the class page.
    assert result.startswith("[_InnerConfig](")
    assert result.endswith(")")


def test_render_type_config_not_in_found():
    # Config type absent from found → backtick name, no link.
    result = _gen.render_type(_InnerConfig, {}, {}, _OWN_PATH)
    assert result == "`_InnerConfig`"


def test_render_type_optional_config():
    result = _gen.render_type(_InnerConfig | None, _FOUND, _CLS_OUTPUT_PATHS, _OWN_PATH)
    assert "[_InnerConfig](" in result
    assert "or `None`" in result


# ---------------------------------------------------------------------------
# render_default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name, expected",
    [
        ("string", '`"hello"`'),
        ("large_int", "`4_294_967_296`"),  # 2**32 with underscores
        ("list_of_str", "`list()`"),
        ("dict_field", "`dict()`"),
    ],
)
def test_render_default_simple(field_name, expected):
    field = _OUTER_FIELDS[field_name]
    resolved = _OUTER_HINTS.get(field_name, field.type)
    assert _gen.render_default(field, resolved, _FOUND) == expected


def test_render_default_none():
    field = _OUTER_FIELDS["inner_optional"]
    assert _gen.render_default(field, _InnerConfig | None, _FOUND) == "`None`"


def test_render_default_required_primitive():
    field = _OUTER_FIELDS["required"]
    assert _gen.render_default(field, str, _FOUND) == "*(required)*"


def test_render_default_config_field_sub_fields_optional():
    # Config-typed field with no default → sub-fields are optional.
    field = _OUTER_FIELDS["inner"]
    assert _gen.render_default(field, _InnerConfig, _FOUND) == "*(sub-fields optional)*"


@config_class()
class _TypeDefaultConfig(Config):
    fmt: type = Field(default=_InnerConfig, hint=FieldHint.core, desc="A type default.")


def test_render_default_type_class():
    # A field whose default value is itself a type object.
    fields = dict(_TypeDefaultConfig.fields())
    assert _gen.render_default(fields["fmt"], type, _FOUND) == "`_InnerConfig`"


# ---------------------------------------------------------------------------
# format_nav_yaml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entries, indent, expected",
    [
        # Flat list of strings.
        (
            ["reference/a.md", "reference/b.md"],
            0,
            ["- reference/a.md", "- reference/b.md"],
        ),
        # Single nested section.
        (
            [{"Section": ["reference/a.md"]}],
            0,
            ["- Section:", "  - reference/a.md"],
        ),
        # Double-nested sections.
        (
            [{"Outer": [{"Inner": ["reference/a.md"]}]}],
            0,
            ["- Outer:", "  - Inner:", "    - reference/a.md"],
        ),
        # Non-zero base indent.
        (
            ["reference/a.md"],
            1,
            ["  - reference/a.md"],
        ),
        # Mixed strings and dicts.
        (
            ["reference/index.md", {"Sub": ["reference/sub/a.md"]}],
            0,
            ["- reference/index.md", "- Sub:", "  - reference/sub/a.md"],
        ),
    ],
)
def test_format_nav_yaml(entries, indent, expected):
    assert _gen.format_nav_yaml(entries, indent) == expected


# ---------------------------------------------------------------------------
# render_class_page smoke test
# ---------------------------------------------------------------------------


def test_render_class_page_contains_key_sections():
    info = _FOUND[_OuterConfig]
    # Build minimal fields list as the generator would.
    fields = []
    for name, field in _OuterConfig.fields():
        if _gen.is_user_field(name, field):
            resolved = _OUTER_HINTS.get(name, field.type)
            fields.append((name, field, resolved))
    info_with_fields = {**info, "fields": fields}

    content = _gen.render_class_page(
        _OuterConfig,
        info_with_fields,
        back_refs=[],
        found=_FOUND,
        cls_output_paths=_CLS_OUTPUT_PATHS,
        own_path=_CLS_OUTPUT_PATHS[_OuterConfig],
    )

    assert "# _OuterConfig" in content
    assert "## Fields" in content
    assert "`string`" in content
    assert "`large_int`" in content
    assert "*(sub-fields optional)*" in content  # inner field
    assert "*(required)*" in content  # required field


# ---------------------------------------------------------------------------
# render_index_page smoke test
# ---------------------------------------------------------------------------


def test_render_index_page_lists_classes():
    classes_in_dir = list(_FOUND.items())
    content = _gen.render_index_page(
        pathlib.Path("tests"),
        classes_in_dir,
        cls_output_paths=_CLS_OUTPUT_PATHS,
        subdirs=[],
    )

    assert "## Classes" in content
    assert "_InnerConfig" in content
    assert "_OuterConfig" in content
