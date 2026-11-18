"""Configuration management for Tally AI Desk."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

APP_NAME = "Tally AI Desk"
APP_VERSION = "1.0.0"

# The whole application is portable: everything resolves relative to this file's parent.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
INBOX_DIR = BASE_DIR / "inbox"
PRESET_DIR = BASE_DIR / "presets"
MODELS_DIR = BASE_DIR / "models"
RUNTIME_DIR = BASE_DIR / "runtime"
TOOLS_DIR = BASE_DIR / "tools"

for _d in (DATA_DIR, LOG_DIR, INBOX_DIR, PRESET_DIR, MODELS_DIR, RUNTIME_DIR, TOOLS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = DATA_DIR / "config.json"
DB_FILE = DATA_DIR / "history.db"

DEFAULT_CONFIG = {
    "tally": {
        "host": "localhost",
        "port": 9000,
        "company": "",
        "gateway_path": "/",
        "timeout_seconds": 15,
        "ledgers_only": True,
        "create_missing_masters": False,
    },
    "ollama": {
        "host": "http://localhost:11434",
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "vision_model": "qwen3-vl:4b",
        "fallback_model": "qwen2.5:1.5b",
        "timeout_seconds": 300,
        "temperature": 0.0,
    },
    "ocr": {
        "render_dpi": 220,
        "max_image_side": 2400,
        "prefer_text_layer": True,
    },
    "app": {
        "theme": "dark",
        "language": "en",
        "auto_check_gateway": True,
        "confirm_before_post": True,
        "save_copies": True,
        "compact_numbers": True,
        "deep_ai_extraction": False,
    },
    "mappings": {
        "default_sales_ledger": "Sales Accounts",
        "default_purchase_ledger": "Purchase Accounts",
        "default_cash_ledger": "Cash",
        "default_bank_ledger": "Bank Accounts",
        "default_gst_ledger": "GST Payable",
        "default_input_gst_ledger": "GST Input Credit",
        "default_discount_ledger": "Discount Received",
        "default_roundoff_ledger": "Round Off",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    def __init__(self, path: Path = CONFIG_FILE):
        self.path = path
        self._lock = threading.RLock()
        self.data = self.load()

    def load(self) -> dict:
        with self._lock:
            if not self.path.exists():
                self.save(DEFAULT_CONFIG)
                data = json.loads(json.dumps(DEFAULT_CONFIG))
            else:
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    data = _deep_merge(DEFAULT_CONFIG, raw)
                except Exception:
                    data = json.loads(json.dumps(DEFAULT_CONFIG))

            # Environment overrides (useful for the portable launcher).
            if os.environ.get("TALLY_AI_OLLAMA_HOST"):
                data["ollama"]["host"] = os.environ["TALLY_AI_OLLAMA_HOST"]
            if os.environ.get("TALLY_AI_OLLAMA_MODEL"):
                data["ollama"]["model"] = os.environ["TALLY_AI_OLLAMA_MODEL"]
            if os.environ.get("TALLY_AI_TALLY_HOST"):
                data["tally"]["host"] = os.environ["TALLY_AI_TALLY_HOST"]
            if os.environ.get("TALLY_AI_TALLY_PORT"):
                try:
                    data["tally"]["port"] = int(os.environ["TALLY_AI_TALLY_PORT"])
                except ValueError:
                    pass
            return data

    def save(self, data: dict | None = None) -> None:
        with self._lock:
            if data is not None:
                self.data = data
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    def get(self, dotted: str, default=None):
        node = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value) -> None:
        with self._lock:
            node = self.data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            self.save()

    def section(self, name: str) -> dict:
        return self.data.get(name, {})


CONFIG = Config()
