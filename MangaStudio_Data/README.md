# Manga Translation Studio

Install the root `requirements.txt` and `requirements-gui.txt`, then run `python MangaStudioMain.py` from the project root. See [完整安装和运行说明](../README_CN.md).

The `app/` directory is desktop application source. Themes, ui_map.json, tasks.json and example dictionaries are shipped resources. Personal profiles, GPT prompt files and schema caches are created locally and excluded from Git. A GPT configuration template is available at `examples/gpt_config-example.yaml` in the project root.

The desktop backend uses the same Python interpreter as the UI process. OpenVINO CLI usage is documented in the root README; the existing GUI retains its original CPU/GPU controls.
