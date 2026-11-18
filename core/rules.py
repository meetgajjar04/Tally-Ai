"""Deterministic invoice parser — fast, accurate rule-based extraction.

For standard Indian GST invoices this is far faster and more accurate than a
small vision LLM: RapidOCR reads the text in ~5s, then these rules build a
balanced voucher instantly. The LLM remains available as a fallback for unusual
documents.
"""
from __future__ import annotations

import re
from datetime import datetime

from .config import CONFIG
from .logger import get_logger
from .vouchers import LedgerEntry, Voucher

log = get_logger("rules")

AMOUNT_RE = re.compile(r"\b(\d[\d,]*\.\d{1,2}|\d[\d,]{2,})\b")
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b", re.I)
DATE_RE = re.compile(
    r"\b(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})\b|\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    re.I,
)
HSN_RE = re.compile(r"\b(\d{4,8})\b")


def _num(text: str) -> float | None:
    cleaned = text.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_amount(line: str) -> float | None:
    """Return the largest plausible amount on a line."""
    amounts = [_num(m.group(1)) for m in AMOUNT_RE.finditer(line)]
    amounts = [a for a in amounts if a is not None and a > 0]
    return max(amounts) if amounts else None


def merge_ocr_lines(lines: list[str]) -> list[str]:
    """Merge OCR label/amount pairs split across separate detection boxes.

    RapidOCR often emits 'CGST @ 9%' then '900.00' as two lines. Joining a
    text-only line with a following pure-number line restores the original
    document layout so the rules can find values on the same line.
    """
    merged: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if (
            i + 1 < len(lines)
            and cur
            and re.search(r"[A-Za-z]", cur)
            and not _find_amount(cur)
            and re.fullmatch(r"[\d,]+\.?\d{0,2}", lines[i + 1].strip())
        ):
            merged.append(f"{cur} {lines[i + 1].strip()}")
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged


def _normalise_date(raw: str) -> str:
    raw = raw.strip()
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d",
        "%d-%m-%y", "%d/%m/%y", "%d %b %Y", "%d %B %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return raw.replace("-", "").replace("/", "").replace(".", "")


def parse_invoice(lines: list[str], source: str = "") -> Voucher | None:
    """Try to build a balanced Sales/Purchase voucher from invoice text lines.

    Returns None when there isn't enough structured data (caller then uses AI).
    """
    full = "\n".join(lines)
    upper = full.upper()

    # --- Invoice number ---
    inv_no = ""
    for pat in [
        r"(?:TAX\s+)?INVOICE\s*(?:NO|NUMBER|#)[\s:.\-]*([A-Z0-9\-/]+)",
        r"BILL\s*(?:NO|NUMBER|#)[\s:.\-]*([A-Z0-9\-/]+)",
        r"\bINV[\s:.\-]*([A-Z0-9\-/]+)",
    ]:
        m = re.search(pat, upper)
        if m:
            inv_no = m.group(1).strip(" .:-")
            break

    # --- Date ---
    date_raw = ""
    dm = DATE_RE.search(full)
    if dm:
        date_raw = dm.group(1) or dm.group(2) or ""
    date_val = _normalise_date(date_raw) if date_raw else datetime.now().strftime("%Y%m%d")

    # --- GSTINs ---
    gstins = [m.group(1).upper() for m in GSTIN_RE.finditer(full)]
    seller_gstin = gstins[0] if gstins else ""
    party_gstin = gstins[1] if len(gstins) > 1 else ""

    # --- Party name ---
    party = ""
    for pat in [
        r"(?:BILL\s+TO|PARTY|CUSTOMER|BUYER|TO)\s*[:\-]\s*(.+)",
        r"(?:BILL\s+TO|PARTY|CUSTOMER|BUYER)\s*\n\s*(.+)",
    ]:
        m = re.search(pat, full, re.I)
        if m:
            candidate = m.group(1).strip().split("\n")[0].strip()
            candidate = re.split(r"\s{2,}|\s+GSTIN|\s+PIN|\s+Phone", candidate, flags=re.I)[0].strip()
            if len(candidate) > 2:
                party = candidate
                break
    if not party and len(gstins) > 1:
        # Fallback: line immediately before the second GSTIN.
        for i, line in enumerate(lines):
            if party_gstin in line.upper() and i > 0:
                party = lines[i - 1].strip()
                break

    # --- Tax lines (CGST / SGST / IGST) ---
    cgst = sgst = igst = 0.0
    gst_rate: float | None = None
    for line in lines:
        u = line.upper()
        amt = _find_amount(line) or 0.0
        rate_m = re.search(r"@\s*(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%", line)
        rate = _num(rate_m.group(1) or rate_m.group(2)) if rate_m else None
        if "CGST" in u:
            cgst = max(cgst, amt)
            gst_rate = gst_rate or (rate * 2 if rate else None)
        elif "SGST" in u:
            sgst = max(sgst, amt)
            gst_rate = gst_rate or (rate * 2 if rate else None)
        elif "IGST" in u:
            igst = max(igst, amt)
            gst_rate = gst_rate or rate
        elif "GST" in u and rate and amt:
            gst_rate = gst_rate or rate

    tax_total = round(cgst + sgst + igst, 2)

    # --- Taxable value & grand total ---
    taxable: float | None = None
    grand_total: float | None = None
    for line in lines:
        u = line.upper()
        amt = _find_amount(line)
        if amt is None:
            continue
        if any(k in u for k in ("TAXABLE VALUE", "TAXABLE AMT", "NET AMOUNT", "SUB TOTAL", "SUBTOTAL")):
            taxable = amt
        if any(k in u for k in ("TOTAL AMOUNT", "GRAND TOTAL", "INVOICE TOTAL", "BILL TOTAL", "AMOUNT DUE")):
            grand_total = amt
        elif u.strip().startswith("TOTAL") and grand_total is None:
            grand_total = amt

    # --- Line items (taxable base) ---
    if taxable is None:
        # Look for a line with HSN code and amounts; the last amount is line total.
        for line in lines:
            if re.match(r"^\d{4,8}\b", line.strip()):
                amounts = [_num(m.group(1)) for m in AMOUNT_RE.finditer(line)]
                amounts = [a for a in amounts if a is not None and a > 0]
                if len(amounts) >= 2:
                    taxable = amounts[-1]
                    break

    if taxable is None and grand_total is not None and tax_total > 0:
        taxable = round(grand_total - tax_total, 2)

    # --- Decide voucher kind ---
    is_purchase = any(k in upper for k in ("PURCHASE BILL", "PURCHASE INVOICE", "BILL FROM", "SUPPLIER", "DEBIT NOTE"))
    if not is_purchase and any(k in upper for k in ("TAX INVOICE", "INVOICE", "SALES")):
        is_purchase = False

    # --- Build voucher only when numbers are coherent ---
    if taxable is None or taxable <= 0:
        return None
    if grand_total is None:
        grand_total = round(taxable + tax_total, 2)
    if abs(round(taxable + tax_total, 2) - round(grand_total, 2)) > 1.0:
        # Numbers don't reconcile — let the AI try.
        log.info("Rule parse skipped: taxable+tax %.2f != total %.2f", taxable + tax_total, grand_total)
        return None
    if not party or len(party) < 2:
        party = "Party"  # user can fix in review

    default_sales = str(CONFIG.get("mappings.default_sales_ledger", "Sales Accounts"))
    default_purchase = str(CONFIG.get("mappings.default_purchase_ledger", "Purchase Accounts"))
    gst_payable = str(CONFIG.get("mappings.default_gst_ledger", "GST Payable"))
    gst_input = str(CONFIG.get("mappings.default_input_gst_ledger", "GST Input Credit"))

    entries: list[LedgerEntry] = []
    if is_purchase:
        entries.append(LedgerEntry(ledger=default_purchase, amount=taxable, dc="Dr", gst_rate=gst_rate, taxable_amount=taxable))
        if cgst:
            entries.append(LedgerEntry(ledger=gst_input, amount=cgst, dc="Dr", gst_rate=(gst_rate / 2 if gst_rate else None), cgst_amount=cgst))
        if sgst:
            entries.append(LedgerEntry(ledger=gst_input, amount=sgst, dc="Dr", gst_rate=(gst_rate / 2 if gst_rate else None), sgst_amount=sgst))
        if igst:
            entries.append(LedgerEntry(ledger=gst_input, amount=igst, dc="Dr", gst_rate=gst_rate, igst_amount=igst))
        entries.append(LedgerEntry(ledger=party, amount=grand_total, dc="Cr", gstin=party_gstin))
        kind = "Purchase"
    else:
        entries.append(LedgerEntry(ledger=party, amount=grand_total, dc="Dr", gstin=party_gstin))
        entries.append(LedgerEntry(ledger=default_sales, amount=taxable, dc="Cr", gst_rate=gst_rate, taxable_amount=taxable))
        if cgst:
            entries.append(LedgerEntry(ledger=gst_payable, amount=cgst, dc="Cr", gst_rate=(gst_rate / 2 if gst_rate else None), cgst_amount=cgst))
        if sgst:
            entries.append(LedgerEntry(ledger=gst_payable, amount=sgst, dc="Cr", gst_rate=(gst_rate / 2 if gst_rate else None), sgst_amount=sgst))
        if igst:
            entries.append(LedgerEntry(ledger=gst_payable, amount=igst, dc="Cr", gst_rate=gst_rate, igst_amount=igst))
        kind = "Sales"

    place = ""
    pm = re.search(r"PLACE\s+OF\s+SUPPLY\s*[:\-]?\s*(.+)", full, re.I)
    if pm:
        place = pm.group(1).strip().split("\n")[0][:60]

    v = Voucher(
        kind=kind,
        date=date_val,
        number=inv_no,
        party=party,
        narration=f"{kind} as per invoice {inv_no or 'N/A'} dated {date_val}",
        entries=entries,
        gst_register="Regular",
        place_of_supply=place,
        is_invoice=bool(inv_no),
        invoice_no=inv_no,
        invoice_date=date_val,
        source_file=source,
        confidence=0.95,
    )
    v.warnings.append("Parsed by fast deterministic rules (review recommended).")
    log.info("Rule parser built %s voucher for %s (total %.2f)", kind, party, grand_total)
    return v
