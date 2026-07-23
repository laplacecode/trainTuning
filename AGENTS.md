# Repository Guidelines

## Project Structure & Module Organization

- `src/trainTuning.App/` contains the .NET 10 WPF application. UI lives in `App.xaml` and `MainWindow.xaml`; MVVM state is under `ViewModels/`, process integration under `Services/`, shared DTOs under `Models/`, and command/binding helpers under `Infrastructure/`.
- `engine/` contains the Python training worker and recommendation rules. `worker.py` is the JSON-lines process entry point; `recommender.py` owns parameter selection.
- `tests/` contains Python unit and worker-protocol tests.
- `scripts/` provides local run and publish automation. `packaging/` contains the Inno Setup definition.
- Generated `bin/`, `obj/`, `artifacts/`, `runs/`, datasets, and model weights are intentionally ignored.

## Architecture Overview

The WPF process owns UI and task lifecycle. It starts a separate Python process and exchanges one JSON object per line over UTF-8 without BOM. Keep training work out of the UI process, preserve backward-compatible field names, and send progress through structured worker events.

## Build, Test, and Development Commands

```powershell
dotnet build trainTuning.slnx
python -m unittest discover -s tests -v
.\scripts\run-dev.ps1
.\scripts\publish.ps1
```

The first command builds the desktop application. The second runs recommendation and protocol tests. `run-dev.ps1` launches the application locally. `publish.ps1` creates a self-contained Windows build under `artifacts/publish/win-x64`.

Install Python dependencies with `python -m pip install -r engine\requirements.txt`. Set `TRAINTUNING_PYTHON` when the desired training interpreter is not on `PATH`.

## Coding Style & Naming Conventions

Use four-space indentation in C# and Python. C# types, properties, and methods use `PascalCase`; private fields use `_camelCase`. Python functions and variables use `snake_case`, with type hints on public functions. Keep nullable reference types enabled and address all build warnings. Reuse WPF resources from `App.xaml` instead of adding one-off button or field styles.

## Testing Guidelines

Use Python’s `unittest`; name files `test_*.py` and methods `test_*`. Add tests for every recommendation-rule change and for protocol encoding or schema changes. There is no fixed coverage threshold, but all tests and a warning-free .NET build are required before review. UI changes should include a result-state screenshot.

## Commit & Pull Request Guidelines

No project-specific commit pattern is established in the available history. Use short, imperative subjects, preferably scoped, such as `feat(engine): add small-object profile` or `fix(ui): unify disabled buttons`. Pull requests should explain behavior changes, list verification commands, link related issues, and include screenshots for WPF changes. Never commit datasets, weights, local Python environments, or credentials.
