"""MkDocs hook: regenerate config reference docs before each build."""

import importlib.util
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
_SCRIPT = _REPO_ROOT / "tools" / "generate_config_docs.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_gen():
    spec = importlib.util.spec_from_file_location("generate_config_docs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def on_pre_build(config) -> None:  # noqa: ANN001
    """Regenerate config reference markdown before the build processes files."""
    gen = _load_gen()
    # Regenerate pages but do not update mkdocs.yaml — nav must be updated
    # manually by running `python tools/generate_config_docs.py` when config
    # classes are added or modules are restructured.
    gen.generate(update_nav=False, verbose=False)
