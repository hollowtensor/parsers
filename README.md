# parsers

Document parsing service with pluggable parsing methods and structured extraction.

## Parsing Methods

| Method | Type | Description |
|--------|------|-------------|
| Docling | Layout-aware | IBM's document understanding library |
| Tesseract | Traditional OCR | Open-source OCR engine |
| DeepSeek OCR | VLM | Vision-language model OCR |
| dots.ocr | VLM | Vision-language model OCR |
| OLM OCR | VLM | Vision-language model OCR |

## Structured Extraction

Extract structured data from parsed documents using:
- **LLM-based extraction** — prompt-driven field extraction
- **DSPy** — programmatic LLM pipelines with Pydantic schema enforcement
- **Pydantic schemas** — type-safe output validation

## Installation

```bash
pip install -e ".[all]"    # everything
pip install -e ".[docling]" # just docling
pip install -e ".[vlm]"     # VLM methods
pip install -e ".[dev]"     # dev tools
```

## Usage

```python
from parsers import parse, extract

# Parse a document
result = parse("invoice.pdf", method="deepseek")

# Structured extraction
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    line_items: list[dict]

invoice = extract(result, schema=Invoice)
```

## Project Structure

```
src/parsers/
├── core/           # Base classes, document model, registry
├── methods/        # Parsing method implementations
│   ├── docling     # Docling parser
│   ├── tesseract   # Tesseract OCR
│   └── vlm/        # Vision-language model parsers
├── extraction/     # Structured extraction (LLM, DSPy, schemas)
└── utils/          # File I/O, preprocessing helpers
```
