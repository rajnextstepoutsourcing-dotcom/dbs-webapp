import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
import fitz  # pymupdf
from dateutil import parser as dateparser


@dataclass
class Field:
    value: str
    confidence: float
    source: str


CERT_RE = re.compile(r"\b(\d{10,14})\b")


def _clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _safe_digits(s: str) -> str:
    s = re.sub(r"\D", "", s or "")
    return s


def _extract_after_label(text: str, label: str) -> str:
    # Looks for: "Label: VALUE" or "Label VALUE" on same line.
    pattern = re.compile(rf"{re.escape(label)}\s*:?\s*(.+)", re.IGNORECASE)
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            return _clean_space(m.group(1))
    return ""


def _parse_dmy(date_str: str) -> Tuple[str, str, str]:
    """Return (day, month, year) as zero-padded strings or ("","","")."""
    if not date_str:
        return "", "", ""
    try:
        dt = dateparser.parse(date_str, dayfirst=True, fuzzy=True)
        if not dt:
            return "", "", ""
        return f"{dt.day:02d}", f"{dt.month:02d}", f"{dt.year:04d}"
    except Exception:
        return "", "", ""


def _extract_from_text(text: str) -> Dict[str, Field]:
    text = text or ""
    out: Dict[str, Field] = {}

    # Certificate Number
    cert_line = _extract_after_label(text, "Certificate Number")
    cert = _safe_digits(cert_line)
    if not cert:
        # sometimes number appears on same line without colon or on next line
        if "Certificate Number".lower() in text.lower():
            # take a window around the phrase
            idx = text.lower().find("certificate number")
            window = text[idx: idx + 200]
            m = CERT_RE.search(window)
            if m:
                cert = m.group(1)
    if cert and 10 <= len(cert) <= 14:
        out["certificate_number"] = Field(cert, 0.90, "PDF text")

    # Surname
    surname = _extract_after_label(text, "Surname")
    surname = re.sub(r"[^A-Za-z\-\s]", "", surname).strip().upper()
    if surname:
        out["surname"] = Field(surname, 0.85, "PDF text")

    # DOB
    dob = _extract_after_label(text, "Date of Birth")
    d, m, y = _parse_dmy(dob)
    if d and m and y:
        out["dob_day"] = Field(d, 0.85, "PDF text")
        out["dob_month"] = Field(m, 0.85, "PDF text")
        out["dob_year"] = Field(y, 0.85, "PDF text")

    # Issue Date (ONLY Date of Issue / Issue Date)
    issue = _extract_after_label(text, "Date of Issue") or _extract_after_label(text, "Issue Date")
    iday, imonth, iyear = _parse_dmy(issue)
    if iday and imonth and iyear:
        out["issue_day"] = Field(iday, 0.85, "PDF text")
        out["issue_month"] = Field(imonth, 0.85, "PDF text")
        out["issue_year"] = Field(iyear, 0.85, "PDF text")

    return out


def _pdf_text(path: Path, max_pages: int = 2) -> str:
    try:
        with pdfplumber.open(str(path)) as pdf:
            texts = []
            for i, page in enumerate(pdf.pages[:max_pages]):
                t = page.extract_text() or ""
                if t.strip():
                    texts.append(t)
            return "\n".join(texts)
    except Exception:
        return ""


def _pdf_to_images_bytes(path: Path, max_pages: int = 2) -> List[bytes]:
    images: List[bytes] = []
    try:
        doc = fitz.open(str(path))
        for i in range(min(max_pages, doc.page_count)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=200)
            images.append(pix.tobytes("png"))
        doc.close()
    except Exception:
        pass
    return images


def _gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # type: ignore

        return genai.Client(api_key=api_key)
    except Exception:
        return None


def _gemini_extract(path: Path) -> Dict[str, Field]:
    client = _gemini_client()
    if not client:
        return {}

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        from google.genai import types  # type: ignore
    except Exception:
        return {}

    parts: List[object] = []
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        b = path.read_bytes()
        if not b:
            return {}
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        parts.append(types.Part.from_bytes(data=b, mime_type=mime))
    elif ext == ".pdf":
        for img in _pdf_to_images_bytes(path, max_pages=2):
            parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))

    if not parts:
        return {}

    prompt = (
        "Extract DBS details from the attached DBS certificate images. "
        "Return ONLY valid JSON with the exact keys below.\n\n"
        "Keys:\n"
        "- certificate_number\n"
        "- surname\n"
        "- dob_day\n"
        "- dob_month\n"
        "- dob_year\n"
        "- issue_day\n"
        "- issue_month\n"
        "- issue_year\n\n"
        "RULES (VERY IMPORTANT):\n"
        "1) Label-based only: extract a value ONLY if it is clearly present near the explicit labels: "
        "'Certificate Number', 'Surname', 'Date of Birth', 'Date of Issue' (or 'Issue Date').\n"
        "2) Do NOT guess or infer. If not clearly visible, return empty string for that key.\n"
        "3) Do NOT invent certificate numbers or dates.\n"
        "4) Certificate number must be digits only (10-14 digits).\n"
        "5) Issue date is ONLY from 'Date of Issue' / 'Issue Date' (ignore Print Date / Printed On).\n"
    )

    try:
        contents = [types.Content(role="user", parts=[types.Part.from_text(prompt)] + parts)]
        resp = client.models.generate_content(model=model, contents=contents)
        text = getattr(resp, "text", "") or ""
        data = _parse_json(text)
        if not isinstance(data, dict):
            return {}

        out: Dict[str, Field] = {}

        cert = _safe_digits(str(data.get("certificate_number", "") or ""))
        if cert and 10 <= len(cert) <= 14:
            out["certificate_number"] = Field(cert, 0.85, f"AI (Gemini Vision: {model})")

        surname = str(data.get("surname", "") or "")
        surname = re.sub(r"[^A-Za-z\-\s]", "", surname).strip().upper()
        if surname:
            out["surname"] = Field(surname, 0.80, f"AI (Gemini Vision: {model})")

        for k in ["dob_day", "dob_month", "dob_year", "issue_day", "issue_month", "issue_year"]:
            v = str(data.get(k, "") or "").strip()
            if v:
                # normalize digits
                v = _safe_digits(v) if "year" in k or "day" in k or "month" in k else v
                out[k] = Field(v.zfill(2) if k.endswith(('_day','_month')) and len(v) <= 2 else v, 0.80, f"AI (Gemini Vision: {model})")

        # If model returned full date strings instead of split, handle minimal
        if not (out.get("dob_day") and out.get("dob_month") and out.get("dob_year")):
            dob_full = str(data.get("dob", "") or "")
            d, m, y = _parse_dmy(dob_full)
            if d and m and y:
                out["dob_day"] = Field(d, 0.80, f"AI (Gemini Vision: {model})")
                out["dob_month"] = Field(m, 0.80, f"AI (Gemini Vision: {model})")
                out["dob_year"] = Field(y, 0.80, f"AI (Gemini Vision: {model})")

        if not (out.get("issue_day") and out.get("issue_month") and out.get("issue_year")):
            issue_full = str(data.get("issue_date", "") or "")
            d, m, y = _parse_dmy(issue_full)
            if d and m and y:
                out["issue_day"] = Field(d, 0.80, f"AI (Gemini Vision: {model})")
                out["issue_month"] = Field(m, 0.80, f"AI (Gemini Vision: {model})")
                out["issue_year"] = Field(y, 0.80, f"AI (Gemini Vision: {model})")

        return out

    except Exception:
        return {}


def _parse_json(text: str):
    text = (text or "").strip()
    # Remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = text.rstrip("`\n ")
    # Find first JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}


def extract_dbs_fields(path: Path) -> Dict[str, object]:
    """Return JSON-safe dict for UI."""
    ext = path.suffix.lower()

    fields: Dict[str, Field] = {}

    if ext == ".pdf":
        text = _pdf_text(path)
        if text.strip():
            fields.update(_extract_from_text(text))

    # If still missing key fields, use vision AI (recommended for scanned PDFs / photos)
    need_ai = False
    for k in ["certificate_number", "surname", "dob_day", "dob_month", "dob_year", "issue_day", "issue_month", "issue_year"]:
        if k not in fields or not (fields[k].value or "").strip():
            need_ai = True
            break

    if need_ai:
        fields_ai = _gemini_extract(path)
        for k, fv in fields_ai.items():
            # only fill missing
            if k not in fields or not fields[k].value:
                fields[k] = fv

    # Build response
    def get_val(key: str) -> str:
        return fields.get(key, Field("", 0.0, "")).value

    def get_conf(key: str) -> float:
        return float(fields.get(key, Field("", 0.0, "")).confidence)

    def get_src(key: str) -> str:
        return fields.get(key, Field("", 0.0, "")).source

    response = {
        "certificate_number": get_val("certificate_number"),
        "surname": get_val("surname"),
        "dob_day": get_val("dob_day"),
        "dob_month": get_val("dob_month"),
        "dob_year": get_val("dob_year"),
        "issue_day": get_val("issue_day"),
        "issue_month": get_val("issue_month"),
        "issue_year": get_val("issue_year"),
        "confidence": {
            "certificate_number": get_conf("certificate_number"),
            "surname": get_conf("surname"),
            "dob": min(get_conf("dob_day"), get_conf("dob_month"), get_conf("dob_year")),
            "issue_date": min(get_conf("issue_day"), get_conf("issue_month"), get_conf("issue_year")),
        },
        "source": {
            "certificate_number": get_src("certificate_number"),
            "surname": get_src("surname"),
            "dob": get_src("dob_day") or get_src("dob_month") or get_src("dob_year"),
            "issue_date": get_src("issue_day") or get_src("issue_month") or get_src("issue_year"),
        },
    }
    return response
