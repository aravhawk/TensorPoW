from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from blake3 import blake3

ROOT = Path(__file__).resolve().parents[2]
WHITEPAPER_DIR = ROOT / "docs" / "whitepaper"
GENESIS_RUNBOOK_PATH = ROOT / "docs" / "genesis" / "ceremony.md"
SOURCE_PATH = WHITEPAPER_DIR / "tensorpow.md"
SCRIPT_PATH = WHITEPAPER_DIR / "build_whitepaper.py"
NORMALIZED_PATH = WHITEPAPER_DIR / "tensorpow.normalized.md"
HASH_PATH = WHITEPAPER_DIR / "tensorpow.blake3"
PDF_PATH = WHITEPAPER_DIR / "tensorpow.pdf"
PDF_HASH_PATH = WHITEPAPER_DIR / "tensorpow.pdf.blake3"
TYPST_PATH = WHITEPAPER_DIR / "tensorpow.typ"


def test_whitepaper_contains_protocol_sections_and_test_evidence() -> None:
    text = SOURCE_PATH.read_text(encoding="utf-8")

    required_sections = [
        "## Abstract",
        "## 1. Introduction",
        "## 3. Cryptographic Commitments",
        "## 4. Tensor Proof of Work",
        "## 5. Fruits, Anchors, and Ordering",
        "## 6. Transactions, Scripts, and UTXO State",
        "## 7. Sharding and Fee Floors",
        "## 8. Relay, Compression, and Data Availability",
        "## 9. Issuance and Fees",
        "## 10. Genesis Commitment",
        "## 11. Security and Determinism Evidence",
    ]
    for section in required_sections:
        assert section in text

    evidence_files = [
        "tests/determinism/test_baseline.py",
        "tests/determinism/test_pow_kernel.py",
        "tests/unit/test_pow_kernel.py",
        "tests/unit/test_ghostdag.py",
        "tests/unit/test_finality.py",
        "tests/unit/test_das.py",
        "tests/unit/test_storage_rocksdb.py",
        "tests/integration/test_rpc_server.py",
    ]
    for evidence_file in evidence_files:
        assert evidence_file in text

    forbidden = ("TO" + "DO", "FIX" + "ME", "place" + "holder")
    for token in forbidden:
        assert token not in text


def test_whitepaper_hash_artifact_is_reproducible() -> None:
    subprocess.run([sys.executable, str(SCRIPT_PATH)], cwd=ROOT, check=True)

    normalized = NORMALIZED_PATH.read_bytes()
    digest = blake3(normalized).hexdigest()
    assert HASH_PATH.read_text(encoding="utf-8") == (f"{NORMALIZED_PATH.name}  blake3  {digest}\n")
    pdf_digest = blake3(PDF_PATH.read_bytes()).hexdigest()
    assert PDF_PATH.read_bytes().startswith(b"%PDF-1.4")
    assert PDF_HASH_PATH.read_text(encoding="utf-8") == f"{PDF_PATH.name}  blake3  {pdf_digest}\n"
    typst = TYPST_PATH.read_text(encoding="utf-8")
    assert typst.startswith("#let") or typst.startswith("#set")
    assert "TensorPoW" in typst


def test_genesis_runbook_uses_pdf_whitepaper_hash_artifact() -> None:
    text = GENESIS_RUNBOOK_PATH.read_text(encoding="utf-8")

    assert "docs/whitepaper/tensorpow.pdf.blake3" in text
    assert "--whitepaper-hash \"$(awk '{print $3}' docs/whitepaper/tensorpow.pdf.blake3)\"" in text
    assert "--whitepaper-hash \"$(awk '{print $3}' docs/whitepaper/tensorpow.blake3)\"" not in text
    assert "--whitepaper-file docs/whitepaper/tensorpow.pdf" not in text
