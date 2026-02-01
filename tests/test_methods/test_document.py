"""Tests for the document model."""

from parsers.core.document import Document, Page, TextBlock, BoundingBox


def test_document_text_property():
    doc = Document(
        source="test.pdf",
        method="test",
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TextBlock(text="Hello"),
                    TextBlock(text="World"),
                ],
            ),
            Page(
                page_number=2,
                blocks=[TextBlock(text="Page 2")],
            ),
        ],
    )
    assert doc.text == "Hello\nWorld\n\nPage 2"
    assert doc.num_pages == 2


def test_bounding_box():
    bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
    block = TextBlock(text="test", bbox=bbox, confidence=0.95)
    assert block.bbox.x1 == 100
    assert block.confidence == 0.95
