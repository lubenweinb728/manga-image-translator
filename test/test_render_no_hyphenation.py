"""Disabling hyphenation must avoid dictionary lookup at every layout stage."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from manga_translator.rendering import text_render, resize_regions_to_font_size


def test_horizontal_layout_skips_dictionary_when_disabled(monkeypatch):
    lookup = Mock(side_effect=AssertionError('Dictionary access is forbidden'))
    monkeypatch.setattr(text_render, 'select_hyphenator', lookup)
    monkeypatch.setattr(text_render, 'get_char_offset_x', lambda size, char: 10)
    lines, widths = text_render.calc_horizontal(20, 'Hello world', 200, 100, hyphenate=False)
    assert ''.join(lines).replace(' ', '') == 'Helloworld'
    assert widths
    lookup.assert_not_called()


def test_region_sizing_passes_hyphenation_setting(monkeypatch):
    layout = Mock(return_value=(['Hello'], [50]))
    monkeypatch.setattr(text_render, 'calc_horizontal', layout)
    rectangle = np.array([[[0, 0], [100, 0], [100, 40], [0, 40]]])
    region = SimpleNamespace(horizontal=True, vertical=False, font_size=20,
                             texts=['Hello'], translation='Hello', text='Hello',
                             unrotated_size=(100, 40), min_rect=rectangle, target_lang='ENG')
    resize_regions_to_font_size(np.zeros((100, 100, 3), np.uint8),
                                [region], None, 0, 1, hyphenate=False)
    assert layout.call_args.kwargs['hyphenate'] is False
