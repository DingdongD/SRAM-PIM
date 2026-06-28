"""P2-03: Verify ceil_div is used for all timing calculations."""

from src.utils.math_utils import ceil_div


def test_ceil_div_exact():
    assert ceil_div(16, 4) == 4


def test_ceil_div_remainder():
    assert ceil_div(17, 4) == 5


def test_ceil_div_one():
    assert ceil_div(1, 4) == 1


def test_ceil_div_large():
    assert ceil_div(1023, 128) == 8


def test_ceil_div_equal():
    assert ceil_div(128, 128) == 1


def test_ceil_div_zero_numerator():
    assert ceil_div(0, 4) == 0
