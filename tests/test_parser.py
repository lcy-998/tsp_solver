from pathlib import Path

import pytest

from tsp.parser import iter_instances, parse_line


def test_parse_coordinates_and_reference_route():
    instance = parse_line("0.1 0.2 0.3 0.4 0.5 0.6 output 2 3 1 2")
    assert instance.coordinates == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    assert instance.reference_route == [2, 3, 1, 2]


def test_parse_without_reference_route():
    assert parse_line("0 0 1 0").reference_route is None


def test_reject_invalid_route():
    with pytest.raises(ValueError):
        parse_line("0 0 1 0 output 1 1")


def test_iter_instances_reads_lines_without_loading_whole_file(tmp_path, monkeypatch):
    path = tmp_path / "instances.txt"
    path.write_text("# comment\n0 0 1 0\n\ninvalid\n", encoding="utf-8")

    def reject_read_text(*args, **kwargs):
        raise AssertionError("iter_instances should not read the whole file")

    monkeypatch.setattr(Path, "read_text", reject_read_text)
    instances = iter_instances(path)
    assert next(instances).name == "instance-1"
    with pytest.raises(ValueError, match=r"instances\.txt:4:"):
        next(instances)

