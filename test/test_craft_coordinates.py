import numpy as np

from manga_translator.detection.default_utils.craft_utils import adjustResultCoordinates


def test_mixed_polygon_sizes_and_missing_polygons():
    quad = np.arange(8, dtype=np.float32).reshape(4, 2)
    curved = np.arange(28, dtype=np.float32).reshape(14, 2)
    original_quad, original_curved = quad.copy(), curved.copy()
    result = adjustResultCoordinates([quad, None, curved], 1.5, 2)
    np.testing.assert_array_equal(result[0], original_quad * [3, 4])
    np.testing.assert_array_equal(result[2], original_curved * [3, 4])
    assert result[1] is None
    np.testing.assert_array_equal(quad, original_quad)
    np.testing.assert_array_equal(curved, original_curved)


def test_rectangular_array_retains_original_behavior():
    boxes = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
    result = adjustResultCoordinates(boxes, 2, 3, ratio_net=1)
    assert isinstance(result, np.ndarray)
    assert result.dtype == boxes.dtype
    np.testing.assert_array_equal(result, boxes * [2, 3])


def test_variable_length_polygons_without_missing_entry():
    values = [np.ones((4, 2), np.float32), np.ones((6, 2), np.float32)]
    result = adjustResultCoordinates(values, .5, .25)
    assert [p.shape for p in result] == [(4, 2), (6, 2)]
    np.testing.assert_array_equal(result[1], np.ones((6, 2)) * [1, .5])


def test_empty_polygons():
    assert adjustResultCoordinates([], 2, 3) == []
