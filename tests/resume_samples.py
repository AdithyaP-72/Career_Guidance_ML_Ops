"""Builders for made-up resumes (PDF / DOCX) used by the resume tests. Nothing here is a real person."""

from __future__ import annotations

import io


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """Smallest valid PDF with selectable text: one list of lines per page. An empty list = a blank (scan-like) page."""
    objs: list[bytes] = []
    n_pages = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    font_obj = 3 + 2 * n_pages
    for i, lines in enumerate(pages):
        content_obj = 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_obj} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>".encode()
        )
        body = "BT /F1 11 Tf 50 750 Td 14 TL\n" + "".join(f"({_esc(x)}) Tj T*\n" for x in lines) + "ET" if lines else ""
        stream = body.encode("latin-1", "replace")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + o + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_docx(paragraphs: list[str], table_rows: list[list[str]] | None = None) -> bytes:
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table_rows:
        t = d.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, cell in enumerate(row):
                t.cell(r, c).text = cell
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# A made-up resume used across tests (ficticious person, ficticious employers).
RESUME_LINES = [
    "Asha Verma",
    "Backend Developer",
    "",
    "SUMMARY",
    "Developer who enjoys clean code. Good communication and management of a small team.",
    "",
    "SKILLS",
    "Python, SQL, ReactJS, Machine Learning, Node JS, C++, R, Tally ERP 9, MS Excel",
    "",
    "EXPERIENCE",
    "Built dashboards with Kubernetes and AWS at Northwind Traders. Core data platform migration.",
]
