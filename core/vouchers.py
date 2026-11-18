"""Pydantic voucher models matching Tally Prime's XML import schema."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VoucherKind = Literal[
    "Sales", "Purchase", "Payment", "Receipt", "Journal", "Contra",
    "Credit Note", "Debit Note", "Sales Return", "Purchase Return",
]


def _norm_date(value: Any) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    s = str(value or "").strip()
    if not s:
        return date.today().strftime("%Y%m%d")
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%Y", "%Y%m%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return s.replace("-", "").replace("/", "").replace(" ", "")


class LedgerEntry(BaseModel):
    ledger: str = Field(..., description="Exact Tally ledger name")
    amount: float = Field(..., ge=0, description="Absolute amount (always positive)")
    dc: Literal["Dr", "Cr"] = Field("Dr", description="Debit or Credit")
    narration: str = ""
    gst_rate: float | None = Field(None, description="GST rate percent, e.g. 18")
    gst_amount: float | None = None
    cgst_amount: float | None = None
    sgst_amount: float | None = None
    igst_amount: float | None = None
    taxable_amount: float | None = None
    hsn_sac: str = ""
    gstin: str = ""
    state: str = ""
    cost_center: str = ""

    @field_validator("ledger", mode="before")
    @classmethod
    def clean_ledger(cls, v: Any) -> str:
        return str(v or "").strip()

    @field_validator("amount", "gst_rate", "gst_amount", "cgst_amount", "sgst_amount", "igst_amount", "taxable_amount", mode="before")
    @classmethod
    def to_float(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(",", "").replace("₹", "").replace("Rs.", "").strip()
        s = s.replace("(", "-").replace(")", "")
        try:
            return float(s)
        except ValueError:
            return 0.0


class Voucher(BaseModel):
    kind: VoucherKind
    date: str = Field(default_factory=lambda: date.today().strftime("%Y%m%d"))
    number: str = ""
    narration: str = ""
    party: str = ""
    entries: list[LedgerEntry] = Field(default_factory=list, min_length=1)
    gst_register: str = "Regular"
    place_of_supply: str = ""
    is_invoice: bool = False
    invoice_no: str = ""
    invoice_date: str = ""
    source_file: str = ""
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    ledger_issues: list[str] = Field(default_factory=list, description="Ledger names not found in Tally")

    @field_validator("date", "invoice_date", mode="before")
    @classmethod
    def normalise_date(cls, v: Any) -> str:
        return _norm_date(v)

    @field_validator("kind", mode="before")
    @classmethod
    def normalise_kind(cls, v: Any) -> Any:
        aliases = {
            "sales": "Sales", "sale": "Sales", "invoice": "Sales",
            "tax invoice": "Sales", "purchase": "Purchase", "bill": "Purchase",
            "payment": "Payment", "receipt": "Receipt", "journal": "Journal",
            "contra": "Contra", "credit note": "Credit Note", "credit": "Credit Note",
            "debit note": "Debit Note", "debit": "Debit Note",
            "sales return": "Sales Return", "purchase return": "Purchase Return",
        }
        s = str(v or "").strip().lower()
        return aliases.get(s, str(v or "Journal").strip().title())

    @model_validator(mode="after")
    def balance_check(self) -> "Voucher":
        total_dr = sum(e.amount for e in self.entries if e.dc == "Dr")
        total_cr = sum(e.amount for e in self.entries if e.dc == "Cr")
        diff = round(total_dr - total_cr, 2)
        if abs(diff) > 0.01:
            self.warnings.append(
                f"Voucher is out of balance by {abs(diff):.2f} (Dr {total_dr:.2f} / Cr {total_cr:.2f}). "
                "Tally may still import it as an unbalanced voucher."
            )
        return self

    @property
    def total(self) -> float:
        return round(sum(e.amount for e in self.entries if e.dc == "Dr"), 2)

    def summary(self) -> str:
        lines = [f"{self.kind} | {_pretty_date(self.date)} | {self.number or '(no number)'} | {self.party or '-'}"]
        lines.append(f"  Narration: {self.narration or '-'}")
        for e in self.entries:
            extra = f" GST {e.gst_rate}%" if e.gst_rate else ""
            lines.append(f"   {e.dc}  {e.ledger:<32} {e.amount:>14,.2f}{extra}")
        if self.warnings:
            lines.append("  ⚠ " + " ".join(self.warnings))
        return "\n".join(lines)


def _pretty_date(yyyymmdd: str) -> str:
    try:
        return datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%d-%m-%Y")
    except Exception:
        return yyyymmdd


class VoucherBatch(BaseModel):
    vouchers: list[Voucher] = Field(default_factory=list)
    source_file: str = ""
    notes: str = ""


def auto_balance(voucher: Voucher, roundoff_ledger: str = "Round Off") -> Voucher:
    """Insert a round-off/T-difference line when the voucher is slightly off."""
    total_dr = round(sum(e.amount for e in voucher.entries if e.dc == "Dr"), 2)
    total_cr = round(sum(e.amount for e in voucher.entries if e.dc == "Cr"), 2)
    diff = round(total_dr - total_cr, 2)
    if abs(diff) <= 0.01:
        return voucher
    if diff > 0:
        voucher.entries.append(LedgerEntry(ledger=roundoff_ledger, amount=diff, dc="Cr", narration="Round off / balancing difference"))
    else:
        voucher.entries.append(LedgerEntry(ledger=roundoff_ledger, amount=abs(diff), dc="Dr", narration="Round off / balancing difference"))
    voucher.warnings.append(f"Auto-balanced with {roundoff_ledger} {abs(diff):.2f}.")
    return voucher
