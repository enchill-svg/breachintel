# Contributing to BreachIntel

Thanks for your interest in contributing to **BreachIntel** — a healthcare breach intelligence and analytics platform. This document outlines how to report issues, propose features, set up a development environment, and submit changes.

---

## How to Report Bugs

- **Search existing issues**: Before opening a new bug, check the [issue tracker](https://github.com/enchill-svg/breachintel/issues) to see if it has already been reported.
- **Use the bug report template**: Click “New issue” → “Bug report” and fill in:
  - Clear description of the problem
  - Steps to reproduce
  - Expected vs. actual behavior
  - Screenshots or logs if available
  - Environment details (OS, Python version, browser, BreachIntel version)
- **Data hygiene**: Do not paste real PHI or sensitive data. Anonymize or synthesize any breach information used in examples.

---

## How to Suggest Features

- **Check existing feature requests** first.
- **Use the feature request template** and include:
  - Problem you’re trying to solve (user story)
  - Proposed solution and desired behavior
  - Alternatives or workarounds you’ve considered
  - Any mockups, diagrams, or references

Well-motivated, narrowly scoped requests are much easier to prioritize and review.

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/breachintel.git
cd breachintel
git remote add upstream https://github.com/enchill-svg/breachintel.git
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

### 4. Optional: Install Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

This runs ruff and other checks automatically before each commit.

---

## Coding Standards

BreachIntel uses **ruff** for linting and formatting (see `[tool.ruff]` in `pyproject.toml`).

- **Style & lint**:

  ```bash
  ruff check .
  ruff format .
  ```

- **Imports**: Keep imports sorted and grouped (standard library, third-party, local).
- **Types**: Prefer type hints for new or modified functions; keep signatures explicit.
- **Docstrings**:
  - Public functions, classes, and modules should have docstrings.
  - Explain *why* and *how*, not just *what* the code does.

Avoid large, monolithic functions. Favor small, testable units with clear responsibilities.

---

## Testing

The test suite is built on **pytest**.

- **Run all tests**:

  ```bash
  python -m pytest
  ```

- **Run a subset**:

  ```bash
  python -m pytest tests/test_cleaner.py -v
  ```

- **Coverage** (CI enforces a minimum threshold):

  ```bash
  python -m pytest --cov=src/breachintel --cov-report=term-missing
  ```

New features and bug fixes should include tests where practical. For UI-only changes, consider adding small smoke tests for the underlying data or logic.

---

## PR Process

1. **Create a feature branch** from `develop` (or `main` if no develop branch is in use):

   ```bash
   git checkout -b feature/my-change
   ```

2. **Make focused commits**:
   - Keep commits small and logically grouped.
   - Avoid unrelated formatting changes in the same PR.

3. **Run checks locally** before pushing:

   ```bash
   ruff check .
   ruff format --check .
   python -m pytest
   ```

4. **Open a Pull Request**:
   - Use the **Pull Request template**.
   - Clearly describe the change, motivation, and any breaking behavior.
   - Link to related issues with `Fixes #123` or `Closes #123` when applicable.

5. **Code Review**:
   - Be responsive to review comments and questions.
   - Prefer follow-up commits over force-push while a PR is under review.
   - Once approved and checks pass, a maintainer will merge the PR.

---

## Thanks

Your contributions — whether bug reports, ideas, docs, or code — help make BreachIntel more useful and reliable for the healthcare security community. Thank you for participating. 🙏

