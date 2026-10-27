"""NLP test dependencies are prepared before workers, never downloaded by tests."""

import importlib
import importlib.metadata
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("installed", [None, "0.0.0"])
def test_pytest_configuration_fails_clearly_without_the_pinned_model(monkeypatch, installed):
    pytest.importorskip("presidio_analyzer")
    from tests import conftest as suite

    original = importlib.metadata.version

    def version(name):
        if name == "en_core_web_lg":
            if installed is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return installed
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    with pytest.raises(pytest.UsageError, match="en_core_web_lg==3.8.0"):
        suite.pytest_configure(SimpleNamespace())


def test_presidio_cannot_start_a_model_download_during_tests(monkeypatch):
    spacy = pytest.importorskip("spacy")
    pytest.importorskip("presidio_analyzer")
    from presidio_analyzer.nlp_engine import SpacyNlpEngine

    downloader = importlib.import_module("spacy.cli.download")

    def network_boundary():
        raise AssertionError("model download reached the network boundary")

    monkeypatch.setattr(downloader, "get_compatibility", network_boundary)
    monkeypatch.setattr(spacy.util, "is_package", lambda model: False)
    with pytest.raises(RuntimeError, match="Runtime NLP model downloads are disabled"):
        SpacyNlpEngine._download_spacy_model_if_needed("missing_test_model")


def test_only_explicit_bootstrap_installs_the_pinned_asset(monkeypatch):
    from tests._helpers import nlp_models

    commands = []
    monkeypatch.setattr(
        nlp_models.subprocess, "run", lambda args, **kwargs: commands.append((args, kwargs))
    )
    nlp_models.main([])
    assert commands == [], "preflight must never install missing assets"
    nlp_models.main(["--install"])
    assert commands == [
        (
            [nlp_models.sys.executable, "-m", "pip", "install", nlp_models.MODEL_WHEEL],
            {"check": True},
        )
    ]
    assert "en_core_web_lg-3.8.0" in nlp_models.MODEL_WHEEL
