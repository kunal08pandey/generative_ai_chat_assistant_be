"""
Document Text Extractor
=======================

Extracts raw text content from uploaded files (PDF, DOCX, TXT, XLSX).
Supports async threading offload at route-level.
"""
from pathlib import Path
import pypdf
import docx
import openpyxl

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_file(file) -> str:
    """
    Extract readable text content from a given FastAPI UploadFile wrapper.
    Supports PDF, DOCX, TXT, and XLSX (Excel) formats.
    """
    suffix = Path(file.filename or "").suffix.lower()
    content_type = file.content_type or ""

    logger.debug("Extracting text from file=%s suffix=%s type=%s", file.filename, suffix, content_type)

    # ── 1. PDF Documents ─────────────────────────────────────────────────
    if content_type == "application/pdf" or suffix == ".pdf":
        reader = pypdf.PdfReader(file.file)
        text_parts = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)

    # ── 2. Word Documents (.docx) ────────────────────────────────────────
    elif (
        content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == ".docx"
    ):
        doc = docx.Document(file.file)
        text_parts = []
        for para in doc.paragraphs:
            if para.text:
                text_parts.append(para.text)
        return "\n".join(text_parts)

    # ── 3. Plain Text Documents (.txt) ───────────────────────────────────
    elif content_type == "text/plain" or suffix == ".txt":
        # Read raw bytes and decode
        data = file.file.read()
        return data.decode("utf-8", errors="replace")

    # ── 4. Excel Spreadsheets (.xlsx) ───────────────────────────────────
    elif suffix in (".xlsx", ".xlsm", ".xltx", ".xltm") or "spreadsheet" in content_type:
        # Load workbook in read_only & data_only mode for maximum speed & memory efficiency
        wb = openpyxl.load_workbook(file.file, read_only=True, data_only=True)
        text_parts = []
        for sheet_name in wb.sheetnames:
            text_parts.append(f"--- Sheet: {sheet_name} ---")
            sheet = wb[sheet_name]
            for row in sheet.iter_rows(values_only=True):
                # Format non-empty cells into comma-separated text values
                row_values = [str(cell) for cell in row if cell is not None]
                if row_values:
                    text_parts.append(", ".join(row_values))
        return "\n".join(text_parts)

    else:
        raise ValueError(
            f"Unsupported file type: {file.filename} (MIME: {content_type}, extension: {suffix})"
        )
