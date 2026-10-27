"""Prepare optional NLP test assets explicitly; workers never install them."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import subprocess
import sys

import pytest

MODEL_NAME = "en_core_web_lg"
MODEL_VERSION = "3.8.0"
MODEL_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
)
_MODEL_HELP = (
    f"NLP tests require {MODEL_NAME}=={MODEL_VERSION}. "
    "Run 'make test-model-install' before pytest; tests never download models."
)


def check_model() -> None:
    if importlib.util.find_spec("presidio_analyzer") is None:
        return
    try:
        installed = importlib.metadata.version(MODEL_NAME)
    except importlib.metadata.PackageNotFoundError:
        raise pytest.UsageError(_MODEL_HELP) from None
    if installed != MODEL_VERSION:
        raise pytest.UsageError(_MODEL_HELP)


def _no_runtime_download(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Runtime NLP model downloads are disabled. " + _MODEL_HELP)


def configure_models(config: pytest.Config) -> None:
    check_model()
    if importlib.util.find_spec("spacy") is None:
        return
    import spacy.cli

    patch = pytest.MonkeyPatch()
    patch.setattr(spacy.cli, "download", _no_runtime_download)
    config.add_cleanup(patch.undo)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install", action="store_true", help="explicitly install the pinned test model"
    )
    args = parser.parse_args(argv)
    if args.install:
        subprocess.run([sys.executable, "-m", "pip", "install", MODEL_WHEEL], check=True)
    else:
        check_model()


if __name__ == "__main__":
    try:
        main()
    except pytest.UsageError as exc:
        raise SystemExit(str(exc)) from None
