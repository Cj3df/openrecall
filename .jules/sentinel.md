## 2024-05-24 - Duplicate Function Definition Causing DoS
**Vulnerability:** A duplicate definition of `record_screenshots_thread` in `openrecall/screenshot.py` shadowed the correct implementation with broken code, causing a runtime crash (DoS) in the recording thread.
**Learning:** Python allows redefinition of functions without warning, making it easy to accidentally paste duplicate code that silently overrides previous definitions.
**Prevention:** Use linters like `flake8` or `pylint` in the CI/CD pipeline to catch `F811` (redefinition of unused name) and `F821` (undefined name) errors before merging.

## 2026-02-15 - DoS via Missing Filename & Strict Validation
**Vulnerability:** A mismatch between the database schema (missing `filename`) and the recorder thread (passing `filename`) caused a crash (DoS). Additionally, strict regex validation in `app.py` blocked access to valid multi-monitor screenshots.
**Learning:** Adding a new column to a critical table requires a backfill strategy to prevent regression for existing data. Strict validation (allowlisting) must be tested against all valid inputs (e.g., generated filenames).
**Prevention:** Use schema migration tools or explicit checks/backfills in `create_db`. Verify validation logic against production data patterns.
