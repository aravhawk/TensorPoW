#!/usr/bin/env python3
"""Reproducibly normalize and hash the TensorPoW whitepaper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

from blake3 import blake3

WHITEPAPER_DIR = Path(__file__).resolve().parent
SOURCE_PATH = WHITEPAPER_DIR / "tensorpow.md"
NORMALIZED_PATH = WHITEPAPER_DIR / "tensorpow.normalized.md"
HASH_PATH = WHITEPAPER_DIR / "tensorpow.blake3"
PDF_PATH = WHITEPAPER_DIR / "tensorpow.pdf"
PDF_HASH_PATH = WHITEPAPER_DIR / "tensorpow.pdf.blake3"
HTML_PATH = WHITEPAPER_DIR / "tensorpow.html"
TYPST_PATH = WHITEPAPER_DIR / "tensorpow.typ"


def normalize_markdown(markdown: str) -> str:
    """Return canonical UTF-8 markdown text with LF endings and one final newline."""
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def build() -> str:
    normalized = normalize_markdown(SOURCE_PATH.read_text(encoding="utf-8"))
    NORMALIZED_PATH.write_text(normalized, encoding="utf-8", newline="\n")
    digest = blake3(normalized.encode("utf-8")).hexdigest()
    HASH_PATH.write_text(
        f"{NORMALIZED_PATH.name}  blake3  {digest}\n",
        encoding="utf-8",
        newline="\n",
    )
    if which("pandoc") is not None:
        _build_html(normalized)
        _build_typst(normalized)
    _build_minimal_pdf(normalized)
    return digest


def _build_html(normalized: str) -> None:
    html = subprocess.run(
        [
            "pandoc",
            "--from",
            "markdown",
            "--to",
            "html5",
            "--standalone",
        ],
        input=normalized.encode("utf-8"),
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    HTML_PATH.write_bytes(html)


def _build_typst(normalized: str) -> None:
    typst = subprocess.run(
        [
            "pandoc",
            "--from",
            "markdown",
            "--to",
            "typst",
            "--standalone",
        ],
        input=normalized.encode("utf-8"),
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    TYPST_PATH.write_bytes(typst)


def _build_minimal_pdf(normalized: str) -> None:
    text = "\n".join(line for line in normalized.splitlines() if line.strip())[:18_000]
    stream = _pdf_text_stream(text)
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
    )
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    PDF_PATH.write_bytes(bytes(output))
    PDF_HASH_PATH.write_text(
        f"{PDF_PATH.name}  blake3  {blake3(PDF_PATH.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
        newline="\n",
    )


def _pdf_text_stream(text: str) -> bytes:
    lines = text.splitlines()
    commands = ["BT", "/F1 8 Tf", "50 760 Td", "10 TL"]
    for line in lines[:70]:
        commands.append(f"({_pdf_escape(line[:110])}) Tj")
        commands.append("T*")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("latin-1")


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def main() -> None:
    try:
        print(build())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing whitepaper PDF tool: {exc.filename}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"whitepaper PDF build failed: {exc}") from exc


if __name__ == "__main__":
    main()
