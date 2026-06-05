from leap import is_leap


def test_divisible_by_4():
    assert is_leap(2024) is True


def test_not_divisible_by_4():
    assert is_leap(2023) is False


def test_century_not_leap():
    assert is_leap(1900) is False


def test_four_hundred_year_leap():
    assert is_leap(2000) is True
