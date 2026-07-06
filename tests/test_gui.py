import tkinter as tk

import pytest

from src.clusterer import DEFAULT_FIXED_K, DEFAULT_K_MAX, DEFAULT_K_MIN
from src.gui import ClusteringApp


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for tkinter")
    instance = ClusteringApp(root)
    yield instance
    root.destroy()


def test_initial_k_values_match_the_shared_constants(app):
    # Guards against the GUI's initial spinbox values silently drifting
    # away from clusterer.py's single source of truth.
    assert app.k_min.get() == DEFAULT_K_MIN
    assert app.k_max.get() == DEFAULT_K_MAX
    assert app.fixed_k.get() == DEFAULT_FIXED_K
