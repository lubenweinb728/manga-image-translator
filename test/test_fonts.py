import hashlib
from unittest.mock import Mock

import pytest

from manga_translator.utils import fonts


def test_font_download_checks_digest_and_reuses_cache(tmp_path, monkeypatch):
    data = b'test font bytes'
    target = tmp_path / 'fonts' / 'font.otf'
    monkeypatch.setattr(fonts, 'DEFAULT_FONT', target)
    monkeypatch.setattr(fonts, 'FONT_SHA256', hashlib.sha256(data).hexdigest())
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [data]
    get = Mock(return_value=response)
    monkeypatch.setattr(fonts.requests, 'get', get)
    assert fonts.ensure_default_font() == str(target)
    assert target.read_bytes() == data
    fonts.ensure_default_font()
    get.assert_called_once()


def test_invalid_font_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    target = tmp_path / 'font.otf'
    target.write_bytes(b'old')
    monkeypatch.setattr(fonts, 'DEFAULT_FONT', target)
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = [b'corrupt']
    monkeypatch.setattr(fonts.requests, 'get', Mock(return_value=response))
    with pytest.raises(RuntimeError, match='--font-path'):
        fonts.ensure_default_font()
    assert target.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [target]
