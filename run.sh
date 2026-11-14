#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    exec "$VIRTUAL_ENV/bin/python" -m manga_translator "$@"
elif [ -x .venv/bin/python ]; then
    exec .venv/bin/python -m manga_translator "$@"
elif [ -x venv/bin/python ]; then
    exec venv/bin/python -m manga_translator "$@"
else
    exec python3 -m manga_translator "$@"
fi
