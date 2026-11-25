"""Inspect decoded prompt data while asserting the structured field exists."""

from defusedxml import ElementTree


def prompt_field(text, field):
    root = ElementTree.fromstring("<inputs>" + text + "</inputs>")
    element = root.find(field)
    assert element is not None, f"Missing prompt field: {field}"
    return element.text or ""
