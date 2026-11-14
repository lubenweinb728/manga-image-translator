"""Download the redistributable default font only when rendering needs it."""
import hashlib
import os
from pathlib import Path
import tempfile

import requests


DEFAULT_FONT = Path(__file__).resolve().parents[2] / 'fonts' / 'NotoSansCJKsc-Regular.otf'
FONT_URL = ('https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/'
            'Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf')
FONT_SHA256 = '2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b'
ARABIC_FONT = DEFAULT_FONT.with_name('NotoSansArabic-Regular.ttf')
ARABIC_URL = ('https://raw.githubusercontent.com/google/fonts/'
              '29b4ec6df2522a8aee279414b78a07910b065874/ofl/notosansarabic/'
              'NotoSansArabic%5Bwdth%2Cwght%5D.ttf')
ARABIC_SHA256 = '63111b5b2e074dd48cc67692e0a2726d86ee94c1c37fe8598257b7b4e87e869e'


def ensure_default_font() -> str:
    return _ensure_font(DEFAULT_FONT, FONT_URL, FONT_SHA256)


def ensure_default_fonts() -> list[str]:
    return [ensure_default_font(), _ensure_font(ARABIC_FONT, ARABIC_URL, ARABIC_SHA256)]


def _ensure_font(path: Path, url: str, expected_hash: str) -> str:
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash:
            return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError('Default font SHA256 mismatch')
        os.replace(temporary, path)
        return str(path)
    except Exception as exc:
        raise RuntimeError(
            f'Cannot prepare default font. Download {url} to {path}, '
            'or supply your own font with --font-path.'
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
