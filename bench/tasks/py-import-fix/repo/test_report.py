from report import render


def test_render_sorted_keys():
    assert render({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
