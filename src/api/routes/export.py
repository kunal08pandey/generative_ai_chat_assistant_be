"""
File & Document Export Endpoint
================================

Provides endpoints to export assistant responses and code into:
- Word documents (.docx) with safe HTML fallback and escaped metadata
- Arbitrary code/text files (.swift, .py, .sh, etc.)
- Excel spreadsheets (.xlsx) with formula injection protection (audit S9)

Phase 1 & Phase 2 Security Improvements:
- Requires authenticated session (`get_current_user`).
- Escapes untrusted HTML in Word HTML fallback to prevent executable markup (audit S5).
- Neutralizes formula injection by quoting text cells starting with [=, +, -, @] (audit S9).
"""
from __future__ import annotations

import csv
import html
import io
import re
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies.auth import get_current_user
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/export", tags=["Export"])


class ExportDocxRequest(BaseModel):
    title: str = Field("Document", max_length=200)
    content: str = Field(..., max_length=500_000)
    filename: str = Field("document.docx", max_length=200)


class ExportFileRequest(BaseModel):
    filename: str = Field(..., max_length=200)
    content: str = Field(..., max_length=500_000)
    mime_type: Optional[str] = None


class ExportXlsxRequest(BaseModel):
    title: str = Field("Sheet1", max_length=100)
    content: str = Field(..., max_length=500_000)
    filename: str = Field("spreadsheet.xlsx", max_length=200)


def extract_table_rows(text: str) -> list[list[str]]:
    """Parse markdown table, CSV, or TSV into a list of row lists."""
    rows: list[list[str]] = []
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]

    # 1. Check if lines contain markdown table pipes (|)
    has_pipe = any("|" in ln for ln in lines)
    if has_pipe:
        for ln in lines:
            if re.match(r"^\|?\s*[-:]+[-|\s:]*\|?$", ln):
                continue
            if "|" in ln:
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if any(cells):
                    rows.append(cells)
        if rows:
            return rows

    # 2. Fallback to CSV / TSV parsing
    delimiter = "\t" if "\t" in text else ","
    try:
        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        for r in reader:
            if r and any(c.strip() for c in r):
                rows.append([c.strip() for c in r])
    except Exception:
        pass

    return rows


@router.post("/docx")
async def export_docx(
    req: ExportDocxRequest,
    user: dict = Depends(get_current_user),
):
    """
    Converts markdown or text into a formatted Word (.docx) document.
    """
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', req.filename)
    if not safe_filename.endswith(".docx"):
        safe_filename += ".docx"

    try:
        import docx
        from docx.shared import Pt, Inches, RGBColor

        doc = docx.Document()

        if req.title:
            doc.add_heading(req.title, level=0)

        lines = req.content.split("\n")
        in_code_block = False
        code_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code_block:
                    code_text = "\n".join(code_lines)
                    p = doc.add_paragraph(code_text)
                    p.paragraph_format.left_indent = Inches(0.25)
                    for r in p.runs:
                        r.font.name = "Courier New"
                        r.font.size = Pt(9.5)
                        r.font.color.rgb = RGBColor(40, 40, 40)
                    code_lines = []
                    in_code_block = False
                else:
                    in_code_block = True
                    code_lines = []
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            if stripped.startswith("# "):
                doc.add_heading(stripped[2:].strip(), level=1)
            elif stripped.startswith("## "):
                doc.add_heading(stripped[3:].strip(), level=2)
            elif stripped.startswith("### "):
                doc.add_heading(stripped[4:].strip(), level=3)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                doc.add_paragraph(stripped[2:].strip(), style="List Bullet")
            elif re.match(r"^\d+\.\s+", stripped):
                clean_text = re.sub(r"^\d+\.\s+", "", stripped)
                doc.add_paragraph(clean_text, style="List Number")
            elif stripped:
                doc.add_paragraph(line)

        if in_code_block and code_lines:
            p = doc.add_paragraph("\n".join(code_lines))
            p.paragraph_format.left_indent = Inches(0.25)
            for r in p.runs:
                r.font.name = "Courier New"
                r.font.size = Pt(9.5)

        bio = io.BytesIO()
        doc.save(bio)
        bio.seek(0)

        return StreamingResponse(
            bio,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )
    except Exception as e:
        logger.warning("python-docx conversion failed (%s). Falling back to HTML .doc format.", e)
        # HTML fallback with strict HTML entity escaping to prevent XSS / markup injection
        safe_title = html.escape(req.title)
        safe_content = html.escape(req.content)
        html_doc = f"""<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head><meta charset='utf-8'><title>{safe_title}</title>
<style>
body {{ font-family: Calibri, sans-serif; font-size: 11pt; line-height: 1.5; color: #1e293b; }}
h1 {{ color: #1e3a8a; font-size: 18pt; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }}
h2 {{ color: #2563eb; font-size: 14pt; margin-top: 16px; }}
pre {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; font-family: Consolas, monospace; font-size: 10pt; }}
</style>
</head><body>
<h1>{safe_title}</h1>
<div><pre>{safe_content}</pre></div>
</body></html>"""
        fallback_filename = safe_filename.replace(".docx", ".doc")
        return Response(
            content=html_doc.encode("utf-8"),
            media_type="application/msword",
            headers={"Content-Disposition": f'attachment; filename="{fallback_filename}"'},
        )


@router.post("/xlsx")
async def export_xlsx(
    req: ExportXlsxRequest,
    user: dict = Depends(get_current_user),
):
    """
    Converts markdown tables or CSV content into a real styled Excel (.xlsx) spreadsheet.
    Protects against Formula Injection (CSV / Excel injection, audit S9).
    """
    rows = extract_table_rows(req.content)
    if not rows:
        rows = [["Item", "Details"], ["Content", req.content]]

    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', req.filename)
    if not safe_filename.endswith(".xlsx"):
        safe_filename += ".xlsx"

    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()
        ws = wb.active
        clean_title = re.sub(r'[\/\\*?:\[\]]', '_', req.title)[:30] or "Sheet1"
        ws.title = clean_title

        header_fill = PatternFill(start_color="107C41", end_color="107C41", fill_type="solid")
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFB", fill_type="solid")
        regular_font = Font(name="Segoe UI", size=10, color="1E293B")

        thin_border = Border(
            left=Side(style="thin", color="CBD5E1"),
            right=Side(style="thin", color="CBD5E1"),
            top=Side(style="thin", color="CBD5E1"),
            bottom=Side(style="thin", color="CBD5E1"),
        )

        for row_idx, row_data in enumerate(rows, start=1):
            for col_idx, cell_value in enumerate(row_data, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                raw_str = str(cell_value).strip()

                # Clean markdown bold/italic tags
                clean_str = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw_str)
                clean_str = re.sub(r"\*([^*]+)\*", r"\1", clean_str)

                # Convert numbers or currency
                num_val = None
                if re.match(r"^-?\d+(\.\d+)?$", clean_str):
                    try:
                        num_val = float(clean_str) if "." in clean_str else int(clean_str)
                    except ValueError:
                        pass
                elif re.match(r"^\$?-?\d{1,3}(,\d{3})*(\.\d+)?$", clean_str):
                    try:
                        clean_num = clean_str.replace("$", "").replace(",", "")
                        num_val = float(clean_num) if "." in clean_num else int(clean_num)
                    except ValueError:
                        pass

                if num_val is not None:
                    cell.value = num_val
                else:
                    # ── Formula Injection Protection (audit S9) ─────────
                    # Neutralize formulas starting with =, +, -, @, \t, \r
                    # by prefixing with a single quote (') so Excel treats it as plain text.
                    if clean_str.startswith(("=", "+", "-", "@", "\t", "\r")):
                        cell.value = f"'{clean_str}"
                    else:
                        cell.value = clean_str

                cell.border = thin_border

                if row_idx == 1:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    ws.row_dimensions[row_idx].height = 28
                else:
                    cell.font = regular_font
                    if row_idx % 2 == 0:
                        cell.fill = zebra_fill
                    cell.alignment = Alignment(
                        horizontal="right" if num_val is not None else "left",
                        vertical="center",
                    )
                    ws.row_dimensions[row_idx].height = 20

        # Auto-adjust column widths
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)

        return StreamingResponse(
            bio,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
        )
    except Exception as e:
        logger.warning("openpyxl export failed (%s). Falling back to CSV.", e)
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        for r in rows:
            # Formula injection protection in CSV fallback
            safe_r = [f"'{c}" if str(c).startswith(("=", "+", "-", "@")) else c for c in r]
            writer.writerow(safe_r)

        content_bytes = b"\xef\xbb\xbf" + csv_buffer.getvalue().encode("utf-8")
        csv_filename = safe_filename.replace(".xlsx", ".csv")
        return Response(
            content=content_bytes,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{csv_filename}"'},
        )


@router.post("/file")
async def export_file(
    req: ExportFileRequest,
    user: dict = Depends(get_current_user),
):
    """
    Downloads any arbitrary code or text file (.swift, .py, .sh, etc.).
    """
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', req.filename) or "generated_file.txt"
    media_type = req.mime_type or "application/octet-stream"
    return Response(
        content=req.content.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
    )
