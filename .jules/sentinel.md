## 2024-05-24 - Duplicate Function Definition Causing DoS
**Vulnerability:** A duplicate definition of `record_screenshots_thread` in `openrecall/screenshot.py` shadowed the correct implementation with broken code, causing a runtime crash (DoS) in the recording thread.
**Learning:** Python allows redefinition of functions without warning, making it easy to accidentally paste duplicate code that silently overrides previous definitions.
**Prevention:** Use linters like `flake8` or `pylint` in the CI/CD pipeline to catch `F811` (redefinition of unused name) and `F821` (undefined name) errors before merging.
