import pytest

from tsp.parser import parse_line


def test_parse_coordinates_and_reference_route():
    instance = parse_line("0.1 0.2 0.3 0.4 0.5 0.6 output 2 3 1 2")
    assert instance.coordinates == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    assert instance.reference_route == [2, 3, 1, 2]


def test_parse_without_reference_route():
    assert parse_line("0 0 1 0").reference_route is None


def test_reject_invalid_route():
    with pytest.raises(ValueError):
        parse_line("0 0 1 0 output 1 1")

