"""Read the real ledger list from the loaded Tally Prime company.

The list is fetched through Tally's XML gateway (no GUI automation) and cached
in data/ledgers.json so the UI still has dropdowns when Tally is temporarily
offline. Every extracted voucher is then mapped onto these exact ledger names
before it can be posted — the app never invents a ledger on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests

from .config import CONFIG, DATA_DIR
from .logger import get_logger
from .tally_xml import gateway_url
from .vouchers import Voucher

log = get_logger("ledgers")

LEDGERS_FILE = DATA_DIR / "ledgers.json"

# Words that do not change the identity of a ledger in Indian Tally companies.
_NOISE = {"a", "c", "ac", "account", "accounts", "ledger", "the", "of", "and"}


def _tag(element: ET.Element) -> str:
    return element.tag.split("}")[-1].upper()


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and common 'A/c / Account' suffixes."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in str(text or "").lower())
    tokens = [tok for tok in cleaned.split() if tok and tok not in _NOISE]
    return " ".join(tokens)


def _child_text(element: ET.Element, tag: str) -> str:
    for child in element:
        if _tag(child) == tag.upper():
            return (child.text or "").strip()
    return ""


def parse_ledgers(xml_text: str) -> list[dict[str, str]]:
    """Parse ledger names from either a TDL collection or a masters export."""
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Tally returned invalid XML: {exc}") from exc

    for element in root.iter():
        if _tag(element) != "LEDGER":
            continue
        name = (element.get("NAME") or element.get("name") or "").strip()
        if not name:
            name = _child_text(element, "NAME")
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        records.append({"name": name, "parent": _child_text(element, "PARENT")})

    records.sort(key=lambda r: r["name"].lower())
    return records


def _export_request() -> str:
    from xml.sax.saxutils import escape

    company = str(CONFIG.get("tally.company", "") or "").strip()
    static = (
        f"<STATICVARIABLES><SVCURRENTCOMPANY>{escape(company)}</SVCURRENTCOMPANY></STATICVARIABLES>"
        if company
        else "<STATICVARIABLES/>"
    )
    return (
        "<ENVELOPE>"
        "<HEADER>"
        "<VERSION>1</VERSION>"
        "<TALLYREQUEST>Export</TALLYREQUEST>"
        "<TYPE>Collection</TYPE>"
        "<ID>LedgerListForAIDesk</ID>"
        "</HEADER>"
        f"<BODY><DESC>{static}"
        "<TDL><TDLMESSAGE>"
        '<COLLECTION NAME="LedgerListForAIDesk" ISINITIALIZATION="Yes">'
        "<TYPE>Ledger</TYPE>"
        "<FETCH>NAME,PARENT</FETCH>"
        "</COLLECTION>"
        "</TDLMESSAGE></TDL>"
        "</DESC><DATA/></BODY>"
        "</ENVELOPE>"
    )


def fetch_ledgers() -> tuple[bool, list[dict[str, str]], str]:
    """Fetch ledgers from Tally. Returns (ok, records, message)."""
    try:
        resp = requests.post(
            gateway_url(),
            data=_export_request().encode("utf-8"),
            headers={"Content-Type": "application/xml", "Accept": "application/xml"},
            timeout=int(CONFIG.get("tally.timeout_seconds", 15)),
            proxies={"http": None, "https": None},
        )
    except Exception as exc:
        return False, [], f"Could not reach Tally to read ledgers: {exc}"
    if resp.status_code != 200:
        return False, [], f"Tally returned HTTP {resp.status_code} while reading ledgers."

    try:
        records = parse_ledgers(resp.text)
    except ValueError as exc:
        return False, [], str(exc)
    if not records:
        return False, [], (
            "Tally responded but no ledgers were found. Make sure a company is loaded "
            "and the gateway is enabled."
        )
    return True, records, f"Found {len(records)} ledgers in Tally."


def save_cache(records: list[dict[str, str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "company": str(CONFIG.get("tally.company", "") or ""),
        "ledgers": records,
    }
    LEDGERS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cache() -> list[dict[str, str]]:
    try:
        data = json.loads(LEDGERS_FILE.read_text(encoding="utf-8"))
        raw = data.get("ledgers", [])
        return [
            {"name": str(item.get("name", "")).strip(), "parent": str(item.get("parent", "")).strip()}
            for item in raw
            if str(item.get("name", "")).strip()
        ]
    except Exception:
        return []


def refresh_ledgers() -> tuple[bool, list[str], str]:
    """Fetch from Tally and update the local cache. Keeps old cache on failure."""
    ok, records, msg = fetch_ledgers()
    if ok:
        save_cache(records)
        log.info("Ledger cache refreshed: %d ledger(s)", len(records))
    return ok, [r["name"] for r in records], msg


def cached_names() -> list[str]:
    return [r["name"] for r in load_cache()]


def match_ledger(name: str, names: list[str] | None = None) -> str | None:
    """Return the exact Tally ledger name closest to *name*, or None."""
    if not name or not str(name).strip():
        return None
    names = list(names) if names is not None else cached_names()
    if not names:
        return None

    target = str(name).strip()
    by_lower = {n.lower(): n for n in names}
    if target.lower() in by_lower:
        return by_lower[target.lower()]

    a = _normalise(target)
    if not a:
        return None
    for candidate in names:
        if _normalise(candidate) == a:
            return candidate

    best: str | None = None
    best_score = 0.0
    tokens_a = set(a.split())
    for candidate in names:
        b = _normalise(candidate)
        if not b:
            continue
        ratio = SequenceMatcher(None, a, b).ratio()
        tokens_b = set(b.split())
        jaccard = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
        score = max(ratio, 0.65 * ratio + 0.35 * jaccard)
        if score > best_score:
            best, best_score = candidate, score
    if best is not None and best_score >= 0.80:
        return best
    return None


@dataclass
class LedgerCheck:
    """Outcome of mapping voucher ledger names onto the Tally ledger list."""

    validated: bool = False
    mapped: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)


def enforce_ledgers(vouchers: list[Voucher], names: list[str] | None = None) -> LedgerCheck:
    """Replace every ledger/party name with an exact existing Tally ledger name.

    Names that cannot be matched are recorded on ``voucher.ledger_issues`` so the
    UI can warn the user and posting can be blocked (when ledgers_only is on).
    """
    check = LedgerCheck()
    names = list(names) if names is not None else cached_names()
    if not names:
        # No cached list (Tally was never reachable) — nothing to validate against.
        return check

    check.validated = True
    for voucher in vouchers:
        voucher.ledger_issues = []
        for entry in voucher.entries:
            match = match_ledger(entry.ledger, names)
            if match:
                if match != entry.ledger:
                    check.mapped.append(f"{entry.ledger} → {match}")
                    entry.ledger = match
            else:
                issue = f"Ledger '{entry.ledger}' does not exist in Tally"
                voucher.ledger_issues.append(issue)
                check.unmapped.append(entry.ledger)
        if voucher.party:
            match = match_ledger(voucher.party, names)
            if match:
                voucher.party = match
            else:
                issue = f"Party ledger '{voucher.party}' does not exist in Tally"
                voucher.ledger_issues.append(issue)
                check.unmapped.append(voucher.party)
        if voucher.ledger_issues:
            voucher.warnings.append(
                "⚠ " + "; ".join(voucher.ledger_issues) +
                ". Create the ledger in Tally (or refresh the ledger list) before posting."
            )
            log.warning(
                "Ledger issues in %s: %s",
                voucher.number or voucher.source_file or "(un-numbered)",
                voucher.ledger_issues,
            )
    if check.mapped:
        log.info("Mapped %d ledger name(s) onto existing Tally ledgers", len(check.mapped))
    return check
