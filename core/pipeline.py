"""End-to-end pipeline: file -> extraction -> AI/rules -> ledger mapping -> Tally import."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

from . import ai, history, ledgers, rules, tally_xml
from .config import CONFIG
from .extractors import extract, is_supported
from .logger import get_logger
from .vouchers import Voucher, auto_balance

log = get_logger("pipeline")

ProgressFn = Callable[[str], None]


@dataclass
class PipelineResult:
    source: str
    vouchers: list[Voucher] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    job_id: int = 0

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.get("success"))

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.get("success"))


def _block_message(v: Voucher) -> str:
    return (
        "Not posted — ledger must already exist in Tally: " + "; ".join(v.ledger_issues) +
        ". Create it in Tally, then use Settings → ↻ Refresh ledgers."
    )


def process_file(
    path: Path | str,
    progress: ProgressFn | None = None,
    post: bool = True,
    create_masters: bool | None = None,
    auto_balance_vouchers: bool = True,
) -> PipelineResult:
    p = Path(path)
    result = PipelineResult(source=p.name)
    emit = progress or (lambda _m: None)

    if not p.exists():
        result.errors.append(f"File not found: {p}")
        return result
    if not is_supported(p):
        result.errors.append(f"Unsupported file type: {p.suffix}")
        return result

    job_id = history.create_job(str(p))
    result.job_id = job_id

    try:
        emit(f"Reading {p.name} …")
        doc = extract(p)
        emit(doc.describe())

        # Ledger names from the real Tally company constrain every extraction path.
        ledger_names = ledgers.cached_names()
        if ledger_names:
            emit(f"Using {len(ledger_names)} existing ledgers from Tally.")
        else:
            emit("No Tally ledger list cached yet — use Settings → ↻ Refresh ledgers.")

        deep_ai = bool(CONFIG.get("app.deep_ai_extraction", False))
        if deep_ai:
            # Thorough mode: read the whole bill with AI before rules get a chance.
            batch = ai.extract_vouchers(doc, progress=emit, ledger_names=ledger_names)
            if batch.notes:
                emit(batch.notes)
            result.vouchers = batch.vouchers

        # Fast path: deterministic rules over the extracted/OCR text.
        if not result.vouchers and doc.text.strip():
            raw_lines = [ln.strip() for ln in doc.text.splitlines() if ln.strip()]
            lines = rules.merge_ocr_lines(raw_lines)
            rule_v = rules.parse_invoice(lines, source=p.name)
            if rule_v:
                emit("Parsed with fast deterministic rules.")
                result.vouchers = [rule_v]

        # Fallback: local Ollama AI model (or primary when deep mode is on and rules failed).
        if not result.vouchers and not (deep_ai and doc.text.strip()):
            batch = ai.extract_vouchers(doc, progress=emit, ledger_names=ledger_names)
            if batch.notes:
                emit(batch.notes)
            result.vouchers = batch.vouchers

        if not result.vouchers:
            detail = "No vouchers could be extracted from this document."
            result.errors.append(detail)
            history.finish_job(job_id, "no_vouchers", detail)
            return result

        for v in result.vouchers:
            if auto_balance_vouchers:
                auto_balance(v, str(CONFIG.get("mappings.default_roundoff_ledger", "Round Off")))
            if not v.narration:
                v.narration = f"Imported from {p.name}"

        # Final safety net: every ledger must be a real ledger in Tally.
        check = ledgers.enforce_ledgers(result.vouchers, ledger_names)
        if check.mapped:
            emit(f"Matched {len(check.mapped)} ledger name(s) to ledgers in Tally.")
        if check.unmapped:
            emit("⚠ Ledger(s) missing in Tally: " + ", ".join(dict.fromkeys(check.unmapped)))

        if post:
            emit(f"Posting {len(result.vouchers)} voucher(s) to Tally …")
            for i, v in enumerate(result.vouchers, 1):
                emit(f"[{i}/{len(result.vouchers)}] Posting {v.kind} {v.number} …")
                if bool(CONFIG.get("tally.ledgers_only", True)) and v.ledger_issues:
                    r: dict[str, Any] = {"voucher": v, "success": False, "message": _block_message(v)}
                else:
                    r = tally_xml.post_voucher(v, create_masters=create_masters)
                result.results.append(r)
                status = "success" if r["success"] else "failed"
                history.add_voucher(job_id, v, status, r["message"])
                emit(f"  → {r['message']}")
            ok = all(r["success"] for r in result.results)
            history.finish_job(
                job_id,
                "success" if ok else "partial_failure",
                f"{result.succeeded}/{len(result.results)} posted",
                len(result.vouchers),
            )
        else:
            for v in result.vouchers:
                result.results.append({"voucher": v, "success": True, "message": "Review mode — not posted.", "dry_run": True})
                history.add_voucher(job_id, v, "review", "Not posted")
            history.finish_job(job_id, "review", "Extracted only", len(result.vouchers))
            emit("Extraction complete. Review the vouchers before posting.")
    except Exception as exc:
        log.exception("Pipeline failed for %s", p.name)
        result.errors.append(str(exc))
        history.finish_job(job_id, "error", str(exc)[:2000])
        emit(f"ERROR: {exc}")

    return result


def post_approved_vouchers(vouchers: list[Voucher], progress: ProgressFn | None = None) -> list[dict[str, Any]]:
    emit = progress or (lambda _m: None)
    out: list[dict[str, Any]] = []

    # Re-check after any manual edits made in the review dialog.
    check = ledgers.enforce_ledgers(vouchers)
    if check.mapped:
        emit(f"Matched {len(check.mapped)} edited ledger name(s) to Tally ledgers.")

    for i, v in enumerate(vouchers, 1):
        emit(f"[{i}/{len(vouchers)}] Posting {v.kind} {v.number} …")
        if bool(CONFIG.get("tally.ledgers_only", True)) and v.ledger_issues:
            r: dict[str, Any] = {"voucher": v, "success": False, "message": _block_message(v)}
        else:
            r = tally_xml.post_voucher(v, create_masters=None)
        out.append(r)
        emit(f"  → {r['message']}")
    return out
