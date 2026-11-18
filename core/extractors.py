"""Document text/table extraction: images, PDFs, Excel workbooks, CSV files.

Everything runs locally. PDFs use their embedded text layer when available and
fall back to rasterising pages with pypdfium2 so Ollama's vision model can OCR them.
"""
from __future__ import annotations

import base64
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import CONFIG
from .logger import get_logger

log = get_logger("extractors")

SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
SUPPORTED_PDF = {".pdf"}
SUPPORTED_EXCEL = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}
SUPPORTED_CSV = {".csv", ".tsv"}
SUPPORTED_TEXT = {".txt", ".json"}
SUPPORTED = SUPPORTED_IMAGE | SUPPORTED_PDF | SUPPORTED_EXCEL | SUPPORTED_CSV | SUPPORTED_TEXT


@dataclass
class ExtractedDoc:
    """Normalised representation handed to the AI layer."""
    name: str
    suffix: str
    text: str = ""
    images: list[bytes] = field(default_factory=list)          # PNG bytes
    tables: list[list[list[Any]]] = field(default_factory=list)  # list of sheets/tables
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_visual(self) -> bool:
        return bool(self.images)

    def compact_text(self, limit: int = 24000) -> str:
        return self.text[:limit]

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.text)} chars text, "
            f"{len(self.images)} page(s)/image(s), {len(self.tables)} table(s)"
        )


def _resize_image(data: bytes) -> bytes:
    """Downscale very large images so the vision model stays fast."""
    from PIL import Image

    max_side = int(CONFIG.get("ocr.max_image_side", 2400))
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) <= max_side:
        return data
    scale = max_side / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _image_to_png(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as img:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    return _resize_image(buf.getvalue())


def _ocr_image(pil_img, name: str = "") -> str:
    """Run local RapidOCR on a PIL image and return joined text lines."""
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR

        engine = _get_ocr_engine()
        result, _elapsed = engine(np.array(pil_img.convert("RGB")))
        if not result:
            log.warning("OCR found no text in %s", name)
            return ""
        lines = [str(item[1]).strip() for item in result if item and len(item) > 1 and str(item[1]).strip()]
        log.info("OCR read %d line(s) from %s", len(lines), name)
        return "\n".join(lines)
    except Exception as exc:
        log.exception("OCR failed for %s: %s", name, exc)
        return ""


_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def extract_image(path: Path, ocr: bool = True) -> ExtractedDoc:
    log.info("Extracting image %s", path.name)
    from PIL import Image

    with Image.open(path) as img:
        pil = img.copy()
    text = _ocr_image(pil, path.name) if ocr else ""
    return ExtractedDoc(
        name=path.name,
        suffix=path.suffix.lower(),
        text=text,
        images=[_image_to_png(path)],
        meta={"kind": "image", "ocr": bool(text)},
    )


def extract_pdf(path: Path) -> ExtractedDoc:
    """Extract embedded text per page; rasterise pages that have little/no text."""
    log.info("Extracting PDF %s", path.name)
    doc = ExtractedDoc(name=path.name, suffix=".pdf", meta={"kind": "pdf"})
    prefer_text = bool(CONFIG.get("ocr.prefer_text_layer", True))

    # 1) Embedded text via pdfplumber
    try:
        import pdfplumber

        chunks: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            doc.meta["pages"] = len(pdf.pages)
            for index, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                table_data = page.extract_tables() or []
                for table in table_data:
                    doc.tables.append(table)
                chunks.append(f"--- PAGE {index + 1} ---\n{page_text}")
                # Keep a raster of every page for vision fallback / side-by-side proof.
                doc.images.append(_render_pdf_page(path, index))
        doc.text = "\n\n".join(chunks).strip()
        if not prefer_text and doc.text:
            doc.images = doc.images  # keep images; AI prompt decides
    except Exception as exc:  # pragma: no cover - depends on installed libs
        log.warning("pdfplumber failed on %s: %s", path.name, exc)
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(path))
            doc.meta["pages"] = len(pdf)
            doc.images = [_render_pdf_page(path, i) for i in range(len(pdf))]
        except Exception as exc2:
            log.error("PDF rasterisation also failed: %s", exc2)
            raise

    # If the text layer is basically empty, OCR the rasterised pages locally.
    if len(doc.text.strip()) < 40 and doc.images:
        doc.meta["needs_vision"] = True
        from PIL import Image
        import io as _io

        ocr_parts: list[str] = []
        for idx, png in enumerate(doc.images, 1):
            pil = Image.open(_io.BytesIO(png))
            ocr_parts.append(f"--- PAGE {idx} (OCR) ---\n" + _ocr_image(pil, f"{doc.name} p{idx}"))
        ocr_text = "\n\n".join(ocr_parts).strip()
        if len(ocr_text) > 40:
            doc.text = ocr_text
            doc.meta["ocr"] = True
            log.info("OCR filled text layer for scanned PDF %s (%d chars)", doc.name, len(ocr_text))
    return doc


def _render_pdf_page(path: Path, index: int) -> bytes:
    import pypdfium2 as pdfium
    from PIL import Image

    dpi = int(CONFIG.get("ocr.render_dpi", 220))
    pdf = pdfium.PdfDocument(str(path))
    page = pdf[index]
    bitmap = page.render(scale=dpi / 72.0)
    pil: Image.Image = bitmap.to_pil()
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    pdf.close()
    return _resize_image(buf.getvalue())


def extract_excel(path: Path) -> ExtractedDoc:
    log.info("Extracting Excel %s", path.name)
    doc = ExtractedDoc(name=path.name, suffix=path.suffix.lower(), meta={"kind": "excel"})
    try:
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        doc.meta["sheets"] = wb.sheetnames
        lines: list[str] = []
        for ws in wb.worksheets:
            table: list[list[Any]] = []
            for row in ws.iter_rows(values_only=True):
                if any(cell is not None and str(cell).strip() != "" for cell in row):
                    table.append(list(row))
            doc.tables.append(table)
            lines.append(f"=== SHEET: {ws.title} ({len(table)} rows) ===")
            for r in table[:400]:
                lines.append(" | ".join("" if c is None else str(c) for c in r))
        wb.close()
        doc.text = "\n".join(lines)
    except Exception as exc:
        log.error("Excel extraction failed for %s: %s", path.name, exc)
        raise
    return doc


def extract_csv(path: Path) -> ExtractedDoc:
    log.info("Extracting CSV %s", path.name)
    doc = ExtractedDoc(name=path.name, suffix=path.suffix.lower(), meta={"kind": "csv"})
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        pass
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    table = [row for row in reader if any(str(c).strip() for c in row)]
    doc.tables.append(table)
    doc.text = "\n".join(" | ".join(row) for row in table[:1000])
    doc.meta["rows"] = len(table)
    return doc


def extract_text(path: Path) -> ExtractedDoc:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    return ExtractedDoc(name=path.name, suffix=path.suffix.lower(), text=raw, meta={"kind": "text"})


def extract(path: Path | str) -> ExtractedDoc:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in SUPPORTED_IMAGE:
        return extract_image(p)
    if suffix in SUPPORTED_PDF:
        return extract_pdf(p)
    if suffix in SUPPORTED_EXCEL:
        return extract_excel(p)
    if suffix in SUPPORTED_CSV:
        return extract_csv(p)
    if suffix in SUPPORTED_TEXT:
        return extract_text(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def images_as_base64(doc: ExtractedDoc, max_pages: int = 8) -> list[str]:
    return [base64.b64encode(img).decode("ascii") for img in doc.images[:max_pages]]


def is_supported(path: Path | str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED
