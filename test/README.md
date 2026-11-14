# Regression tests

Install `requirements.txt`, `requirements-dev.txt`, and optionally `requirements-openvino.txt`. Run `python -m pytest test` from the project root.

Translator requests are mocked. First use of tiktoken can fetch its public vocabulary cache. Rendering tests download the default Noto font if absent; see `fonts/README.md` for offline preparation. Test outputs and caches are not part of the release.
