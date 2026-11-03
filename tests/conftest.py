import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test runs against a scratch KOBOX_HOME (no models, no audio device)."""
    home = tmp_path / "kobox"
    monkeypatch.setenv("KOBOX_HOME", str(home))
    monkeypatch.setenv("KOBOX_NO_SEED", "1")          # tests start from an empty bank unless they seed one
    import kobox.paths
    importlib.reload(kobox.paths)
    for m in list(sys.modules):
        if m.startswith("kobox.") and m != "kobox.paths":
            importlib.reload(sys.modules[m])
    kobox.paths.ensure_dirs()
    return home
