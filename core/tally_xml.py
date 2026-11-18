"""Tally Prime XML HTTP gateway client.

Posts voucher imports to http://localhost:9000 and parses Tally's success/failure
response. No GUI automation, no keystrokes — accurate and repeatable.

Schema follows Tally Prime's Data Interchange format:
  ENVELOPE > HEADER (Import/Data/Vouchers), BODY > DESC (STATICVARIABLES) + DATA (TALLYMESSAGE)
  Voucher legs use <ALLLEDGERENTRIES.LIST> with <ISDEEMEDPOSITIVE> and signed amounts.
  NOTE (counter-intuitive but required by Tally): debit legs are NEGATIVE, credit legs POSITIVE.
"""
from __future__ import annotations

import re
from typing import Any
from xml.sax.saxutils import escape

import requests

from .config import CONFIG
from .logger import get_logger
from .vouchers import Voucher

log = get_logger("tally_xml")


def _company() -> str:
    return str(CONFIG.get("tally.company", "") or "").strip()


def _envelope(body: str, import_id: str = "Vouchers") -> str:
    """Wrap TALLYMESSAGE records in the standard Tally import envelope."""
    company = _company()
    static = (
        f"<STATICVARIABLES><SVCURRENTCOMPANY>{escape(company)}</SVCURRENTCOMPANY></STATICVARIABLES>"
        if company else "<STATICVARIABLES/>"
    )
    return (
        "<ENVELOPE>"
        "<HEADER>"
        "<VERSION>1</VERSION>"
        "<TALLYREQUEST>Import</TALLYREQUEST>"
        "<TYPE>Data</TYPE>"
        f"<ID>{escape(import_id)}</ID>"
        "</HEADER>"
        "<BODY>"
        f"<DESC>{static}</DESC>"
        "<DATA>"
        f"{body}"
        "</DATA>"
        "</BODY>"
        "</ENVELOPE>"
    )


def build_voucher_xml(v: Voucher) -> str:
    """Build a Tally-compatible Voucher import message."""
    # Tally needs plain YYYYMMDD for <DATE> (some builds accept YYYYMMDDHHMMSS).
    raw_date = re.sub(r"\D", "", str(v.date))
    date_of_voucher = raw_date[:8] if len(raw_date) >= 8 else raw_date

    number = str(v.number or "").strip()
    narration = str(v.narration or f"Imported via Tally AI Desk | {v.source_file}").strip()
    party = str(v.party or "").strip()
    kind = str(v.kind).strip()

    lines: list[str] = [
        f'<VOUCHER VCHTYPE="{escape(kind)}" ACTION="Create">',
        f"<DATE>{date_of_voucher}</DATE>",
        f"<VOUCHERTYPENAME>{escape(kind)}</VOUCHERTYPENAME>",
    ]
    if number:
        lines.append(f"<VCHNO>{escape(number)}</VCHNO>")
    if party:
        lines.append(f"<PARTYLEDGERNAME>{escape(party)}</PARTYLEDGERNAME>")
    lines.append(f"<NARRATION>{escape(narration)}</NARRATION>")
    if str(v.gst_register or "Regular"):
        lines.append(f"<GSTREGISTER>{escape(str(v.gst_register or 'Regular'))}</GSTREGISTER>")
    if v.is_invoice and v.invoice_no:
        lines.append(f"<INVOICENO>{escape(v.invoice_no)}</INVOICENO>")
    if v.invoice_date:
        inv = re.sub(r"\D", "", str(v.invoice_date))[:8]
        if inv:
            lines.append(f"<INVOICEDATE>{inv}</INVOICEDATE>")
    if v.place_of_supply:
        lines.append(f"<PLACEOFSUPPLY>{escape(v.place_of_supply)}</PLACEOFSUPPLY>")

    for e in v.entries:
        amount = abs(float(e.amount))
        # Tally semantics: negative amount = Debit, positive = Credit.
        signed = f"-{amount:.2f}" if e.dc == "Dr" else f"{amount:.2f}"
        deemed = "Yes" if e.dc == "Dr" else "No"
        is_party = "Yes" if party and e.ledger.strip().lower() == party.lower() else "No"
        lines.append("<ALLLEDGERENTRIES.LIST>")
        lines.append(f"<LEDGERNAME>{escape(e.ledger)}</LEDGERNAME>")
        lines.append(f"<ISDEEMEDPOSITIVE>{deemed}</ISDEEMEDPOSITIVE>")
        lines.append(f"<AMOUNT>{signed}</AMOUNT>")
        lines.append(f"<ISPARTYLEDGER>{is_party}</ISPARTYLEDGER>")
        if e.gst_rate is not None:
            lines.append(f"<GSTRATE>{float(e.gst_rate):g}</GSTRATE>")
        if e.gstin:
            lines.append(f"<PARTYGSTIN>{escape(e.gstin)}</PARTYGSTIN>")
        if e.state:
            lines.append(f"<GSTSTATE>{escape(e.state)}</GSTSTATE>")
        if e.hsn_sac:
            lines.append(f"<HSNSACCODE>{escape(e.hsn_sac)}</HSNSACCODE>")
        if e.cgst_amount is not None:
            lines.append(f"<CGSTAMOUNT>{float(e.cgst_amount):.2f}</CGSTAMOUNT>")
        if e.sgst_amount is not None:
            lines.append(f"<SGSTAMOUNT>{float(e.sgst_amount):.2f}</SGSTAMOUNT>")
        if e.igst_amount is not None:
            lines.append(f"<IGSTAMOUNT>{float(e.igst_amount):.2f}</IGSTAMOUNT>")
        if e.cost_center:
            lines.append(f"<COSTCENTRE>{escape(e.cost_center)}</COSTCENTRE>")
        lines.append("</ALLLEDGERENTRIES.LIST>")

    lines.append("</VOUCHER>")
    return _envelope(f'<TALLYMESSAGE xmlns:UDF="TallyUDF">{"".join(lines)}</TALLYMESSAGE>')


def build_master_xml(names: list[str]) -> str:
    """Create Ledger masters before importing vouchers that reference them."""
    lines: list[str] = []
    seen: set[str] = set()
    for raw in names:
        clean = str(raw).strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        lines.append(
            f'<LEDGER NAME="{escape(clean)}" ACTION="Create">'
            f"<NAME>{escape(clean)}</NAME>"
            "</LEDGER>"
        )
    if not lines:
        return ""
    return _envelope(f'<TALLYMESSAGE xmlns:UDF="TallyUDF">{"".join(lines)}</TALLYMESSAGE>', "All Masters")


def gateway_url() -> str:
    host = str(CONFIG.get("tally.host", "localhost")).strip()
    port = int(CONFIG.get("tally.port", 9000))
    path = str(CONFIG.get("tally.gateway_path", "/") or "/")
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{host}:{port}{path}"


def check_gateway() -> tuple[bool, str]:
    """Probe with a harmless export request."""
    probe = (
        "<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Export</TALLYREQUEST>"
        "<TYPE>Data</TYPE><ID>List of Companies</ID></HEADER>"
        "<BODY><DESC><STATICVARIABLES/></DESC><DATA/></BODY></ENVELOPE>"
    )
    try:
        resp = requests.post(
            gateway_url(),
            data=probe.encode("utf-8"),
            headers={"Content-Type": "application/xml"},
            timeout=5,
            proxies={"http": None, "https": None},
        )
        if resp.status_code == 200:
            return True, f"Tally gateway online at {gateway_url()}"
        return False, f"Tally gateway returned HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return False, f"Tally gateway unreachable at {gateway_url()}: {exc}"


def _parse_tally_result(xml_text: str) -> tuple[bool, str]:
    lower = xml_text.lower()
    if any(tok in lower for tok in ("<created>", "successfully created", "created successfully", "voucher created")):
        return True, "Created successfully"
    if any(tok in lower for tok in ("<failed>", "not created", "error in import", "invalid")):
        m = re.search(r"<(?:Message|Description|Status|Error)>(.*?)</(?:Message|Description|Status|Error)>", xml_text, re.I | re.S)
        detail = m.group(1).strip() if m else re.sub(r"<[^>]+>", " ", xml_text)[:300]
        return False, detail
    if "<envelope" in lower:
        # Tally usually returns an empty/ack envelope on successful import.
        return True, "Accepted by Tally"
    return False, f"Unknown Tally response: {xml_text[:200]}"


def post_xml(xml_payload: str) -> tuple[bool, str]:
    timeout = int(CONFIG.get("tally.timeout_seconds", 15))
    try:
        resp = requests.post(
            gateway_url(),
            data=xml_payload.encode("utf-8"),
            headers={"Content-Type": "application/xml", "Accept": "application/xml"},
            timeout=timeout,
            proxies={"http": None, "https": None},
        )
        log.debug("Tally response %s: %s", resp.status_code, resp.text[:500])
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        return _parse_tally_result(resp.text)
    except Exception as exc:
        log.exception("Tally post failed: %s", exc)
        return False, str(exc)


def post_voucher(v: Voucher, create_masters: bool | None = None, dry_run: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"voucher": v, "success": False, "message": "", "dry_run": dry_run}
    xml = build_voucher_xml(v)
    result["xml"] = xml

    if dry_run:
        result["success"] = True
        result["message"] = "Dry run — not sent to Tally."
        return result

    if bool(CONFIG.get("tally.ledgers_only", True)) and getattr(v, "ledger_issues", None):
        result["message"] = (
            "Not posted — ledger must already exist in Tally: " + "; ".join(v.ledger_issues)
        )
        return result

    if create_masters is None:
        create_masters = bool(CONFIG.get("tally.create_missing_masters", False))
    if create_masters:
        ledger_names = [e.ledger for e in v.entries] + ([v.party] if v.party else [])
        master_xml = build_master_xml(ledger_names)
        if master_xml:
            master_ok, master_msg = post_xml(master_xml)
            log.info("Master creation: %s — %s", master_ok, master_msg)

    ok, msg = post_xml(xml)
    result["success"] = ok
    result["message"] = msg
    return result
