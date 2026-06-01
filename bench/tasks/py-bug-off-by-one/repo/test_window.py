from window import window


def test_first_three():
    assert window([1, 2, 3, 4, 5], 3) == [1, 2, 3]


def test_zero_returns_empty():
    assert window([1, 2, 3], 0) == []


def test_n_equal_len():
    assert window([1, 2], 2) == [1, 2]
