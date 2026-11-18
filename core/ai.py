"""Ollama integration: turn extracted documents into structured vouchers."""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import CONFIG
from .extractors import ExtractedDoc, images_as_base64
from .logger import get_logger
from .vouchers import VoucherBatch

log = get_logger("ai")

SYSTEM_PROMPT_BASE = """You are an expert Indian accounting data-extraction assistant. You read the ENTIRE invoice, bill,
receipt, payment advice, GST tax invoice, credit/debit note or spreadsheet before writing a single entry.
You output STRICT JSON only.

Your job: convert every financial document into Tally Prime voucher entries.

RULES:
1. Return ONLY valid JSON matching the schema below. No markdown, no commentary, no code fences.
2. Read the whole document: header, seller/buyer blocks, GSTINs, every line item, every tax line,
   discounts, freight, round-off and the grand total. Do not stop at the first number you see.
   Use the ledger names supplied in the ALLOWED LEDGER LIST when one is provided — see the
   LEDGER RULE appended below. Without such a list, use names from the document, and for generic
   heads use: Sales Accounts, Purchase Accounts, Cash, Bank Accounts, GST Payable, GST Input Credit,
   Discount Received, Round Off, Freight Outward, Freight Inward.
3. Every entry has "ledger", "amount" (positive number, never negative), "dc" ("Dr" or "Cr").
4. A voucher MUST balance: total Dr must equal total Cr. If a small rounding gap exists (< 1% of total),
   add a "Round Off" entry to balance it. Never invent large amounts.
5. GST handling:
   - Sales/tax invoice: party A/c Dr (total invoice value); Sales Accounts Cr (taxable value);
     GST Payable Cr (CGST+SGST or IGST).
   - Purchase bill: Purchase Accounts Dr (taxable); GST Input Credit Dr (tax); party A/c Cr (total).
   - If CGST and SGST are given separately, put each in its own entry with gst_rate and gst_amount.
     If IGST is given, use one IGST entry. Set "gst_register" to "Regular".
   - Extract GSTIN, HSN/SAC and place of supply when visible.
6. Voucher kinds: Sales, Purchase, Payment, Receipt, Journal, Contra, Credit Note, Debit Note,
   Sales Return, Purchase Return.
7. Dates: use YYYY-MM-DD format from the document. If absent, use today's date.
8. "number" = the document's invoice/bill/voucher number. Set "is_invoice": true when a tax invoice number exists.
9. "confidence" = 0.0-1.0 how sure you are about the whole voucher.
10. If one file contains multiple documents/vouchers, output all of them in "vouchers".
11. Never guess amounts that are not in the document. Prefer fewer, accurate vouchers over invented ones.
12. If the document is unreadable, return {"vouchers": [], "notes": "reason"}.

JSON SCHEMA:
{
  "vouchers": [
    {
      "kind": "Sales",
      "date": "2026-11-18",
      "number": "INV-001",
      "party": "Customer Name",
      "narration": "Against invoice INV-001 dated 18-11-2026",
      "gst_register": "Regular",
      "place_of_supply": "Maharashtra - 27",
      "is_invoice": true,
      "invoice_no": "INV-001",
      "invoice_date": "2026-11-18",
      "confidence": 0.95,
      "entries": [
        {"ledger": "Customer Name", "amount": 11800.0, "dc": "Dr", "gstin": "27AAPFU0939F1ZV", "state": "Maharashtra"},
        {"ledger": "Sales Accounts", "amount": 10000.0, "dc": "Cr", "gst_rate": 18, "taxable_amount": 10000.0, "hsn_sac": "998314"},
        {"ledger": "GST Payable", "amount": 1800.0, "dc": "Cr", "gst_rate": 18, "gst_amount": 1800.0, "cgst_amount": 900.0, "sgst_amount": 900.0}
      ]
    }
  ],
  "notes": "any caveats"
}"""


LEDGER_RULE = """

===== LEDGER RULE — HIGHEST PRIORITY =====
The user's Tally company already contains the ledgers listed below. You MUST use ONLY these
ledger names, copied EXACTLY as written (same spelling, spacing and capitalisation).
Do NOT invent, abbreviate, translate or "improve" a ledger name. Do NOT append "A/c" or "Account".
Never create a ledger that is not in this list.

How to choose:
- Read the whole bill first, then pick the listed ledger that best represents each role:
  the party/customer/supplier, the sales or purchase head, and each GST head (CGST/SGST/IGST).
- A party name on the bill may appear in Tally with a slightly different spelling — still use the
  exact listed name and put the bill's original party name in "narration" or "notes".
- If the bill mentions a party with no matching ledger, use the most suitable existing generic
  party ledger from the list (for example Sundry Debtors / Sundry Creditors) and record the
  original party name in "notes".
- If NO ledger in the list can represent a required entry, do not invent one: omit that voucher
  and explain the missing ledger in "notes" so a human can create it in Tally first.

ALLOWED LEDGERS IN TALLY ({count} total):
{ledger_list}
===== END LEDGER RULE =====
"""


def system_prompt(ledger_names: list[str] | None = None) -> str:
    """Build the system prompt, adding the live Tally ledger whitelist when available."""
    if not ledger_names:
        return SYSTEM_PROMPT_BASE
    unique: list[str] = []
    seen: set[str] = set()
    for name in ledger_names:
        clean = str(name).strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            unique.append(clean)
    listing = "\n".join(f"- {name}" for name in unique[:800])
    return SYSTEM_PROMPT_BASE + LEDGER_RULE.format(count=len(unique), ledger_list=listing)


def _clean_json(text: str) -> str:
    text = text.strip()
    # Strip markdown fences some models add despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0] or [0])
    if text.startswith("[") and start == 0:
        text = '{"vouchers": ' + text + "}"
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        text = text[first:last + 1]
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _ollama_url(path: str) -> str:
    host = str(CONFIG.get("ollama.host", "http://localhost:11434")).rstrip("/")
    return host + path


def check_ollama() -> tuple[bool, str]:
    try:
        resp = requests.get(_ollama_url("/api/tags"), timeout=5, proxies={"http": None, "https": None})
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return True, "Ollama online. Models: " + ", ".join(models[:8])
    except Exception as exc:
        return False, f"Ollama unreachable: {exc}"


def list_models() -> list[str]:
    try:
        resp = requests.get(_ollama_url("/api/tags"), timeout=5, proxies={"http": None, "https": None})
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return []


def _supports_vision(model: str) -> bool:
    """Ask Ollama whether a model can accept image input."""
    try:
        resp = requests.post(
            _ollama_url("/api/show"),
            json={"name": model},
            timeout=10,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        info = resp.json()
        return "vision" in (info.get("capabilities") or []) or bool(info.get("projectors"))
    except Exception:
        return False


def pick_model_for(doc: ExtractedDoc, requested: str | None = None) -> str:
    """Choose a model that can handle this document.

    Images / scanned PDFs require a vision-capable model. If the configured model
    is text-only, automatically fall back to an available vision model so the
    user does not have to diagnose a 500 error.
    """
    requested = requested or str(CONFIG.get("ollama.model", ""))
    if not doc.has_visual:
        return requested
    if _supports_vision(requested):
        return requested

    available = list_models()
    configured_vision = str(CONFIG.get("ollama.vision_model", "qwen3-vl:4b") or "")
    preferred = []
    if configured_vision and configured_vision in available:
        preferred.append(configured_vision)
    preferred += [
        m for m in available
        if m not in preferred
        and any(k in m.lower() for k in ("qwen3-vl", "qwen2-vl", "llava", "bakllava", "vision", "gemma3", "qwen2.5-vl"))
    ]
    for candidate in preferred:
        if _supports_vision(candidate):
            log.info("Auto-switching to vision model %s for %s", candidate, doc.name)
            return candidate
    # No vision model available — return requested so the caller reports a clear error.
    log.warning("No vision model available for %s (images present)", doc.name)
    return requested


def _chat(payload: dict) -> dict:
    timeout = int(CONFIG.get("ollama.timeout_seconds", 300))
    resp = requests.post(
        _ollama_url("/api/chat"),
        json=payload,
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    return resp.json()


MAX_PROMPT_CHARS = 120_000


def _full_document_text(doc: ExtractedDoc) -> str:
    """Give the model the WHOLE bill: all text plus every row of every table."""
    chunks: list[str] = []
    if doc.text:
        chunks.append(doc.text.strip())
    for table in doc.tables:
        for row in table:
            chunks.append(" | ".join("" if cell is None else str(cell) for cell in row))
        chunks.append("--- end of table ---")
    return "\n".join(chunk for chunk in chunks if chunk)[:MAX_PROMPT_CHARS]


def extract_vouchers(
    doc: ExtractedDoc,
    model: str | None = None,
    progress: Any = None,
    retry: bool = True,
    ledger_names: list[str] | None = None,
) -> VoucherBatch:
    """Run the vision/text model over one extracted document.

    When *ledger_names* is supplied (the live Tally ledger list), the model is
    told to read the entire document and use only those exact ledger names.
    """
    requested = model or str(CONFIG.get("ollama.model", "qwen3:4b-instruct-2507-q4_K_M"))
    model = pick_model_for(doc, requested)
    if doc.has_visual and not _supports_vision(model):
        return VoucherBatch(
            vouchers=[],
            source_file=doc.name,
            notes=(
                "This document contains images but the selected AI model cannot read images. "
                "Install a vision model with:  ollama pull qwen3-vl:4b  "
                "then select it in Settings → Model for documents."
            ),
        )
    image_b64: list[str] = []

    # Full text + every table row (no 120-row truncation) so nothing on the bill is missed.
    content_hint = _full_document_text(doc)
    prompt = (
        f"Read the ENTIRE document below and extract every accounting voucher: {doc.name}\n"
        f"File type: {doc.suffix}\n"
        "Pay attention to the party block, all line items, every tax line, discounts, "
        "freight, round-off and the grand total.\n\n"
        f"FULL DOCUMENT CONTENT:\n{content_hint or '(no text layer — read the attached image(s))'}\n\n"
    )
    if ledger_names:
        allowed = "\n".join(f"- {name}" for name in ledger_names[:800])
        prompt += (
            "USE ONLY the following existing Tally ledger names, exactly as written: \n"
            f"{allowed}\n\n"
        )
    else:
        allowed = ""
    prompt += (
        "Return the STRICT JSON described in your instructions. "
        "If this is an image/PDF scan, read the values visually from the attached page image(s)."
    )
    if doc.has_visual:
        image_b64 = images_as_base64(doc, max_pages=6)

    # Ollama's /api/chat expects plain string content; images go in the message's
    # separate "images" array (base64, no data-URL prefix).
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_b64:
        user_message["images"] = image_b64

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt(ledger_names)},
            user_message,
        ],
        "options": {
            "temperature": float(CONFIG.get("ollama.temperature", 0.0)),
            "num_predict": 4096,
        },
    }

    if progress:
        progress(f"Sending {doc.name} to {model} …")
    log.info(
        "Calling Ollama model=%s for %s (images=%d, chars=%d, allowed_ledgers=%d)",
        model, doc.name, len(doc.images), len(content_hint), len(ledger_names or []),
    )

    try:
        result = _chat(payload)
        raw = result.get("message", {}).get("content", "")
        parsed = json.loads(_clean_json(raw))
        batch = VoucherBatch.model_validate(parsed)
        for v in batch.vouchers:
            v.source_file = doc.name
        batch.source_file = doc.name
        if progress:
            progress(f"AI extracted {len(batch.vouchers)} voucher(s) from {doc.name}.")
        return batch
    except Exception as exc:
        log.exception("AI extraction failed for %s: %s", doc.name, exc)
        if retry and model != CONFIG.get("ollama.fallback_model"):
            fallback = str(CONFIG.get("ollama.fallback_model", "qwen2.5:1.5b"))
            if fallback in list_models():
                if progress:
                    progress(f"Retrying with fallback model {fallback} …")
                return extract_vouchers(
                    doc, model=fallback, progress=progress, retry=False,
                    ledger_names=ledger_names,
                )
        if progress:
            progress(f"AI extraction failed for {doc.name}: {exc}")
        return VoucherBatch(vouchers=[], source_file=doc.name, notes=f"AI error: {exc}")
