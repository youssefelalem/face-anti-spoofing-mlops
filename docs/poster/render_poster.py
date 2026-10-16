"""Exporte le poster en PDF et en PNG avec PowerPoint (rendu fidele aux polices).

Utilise une liaison COM dynamique (pywin32), qui ne depend pas de la bibliotheque de types
d'Office (parfois mal enregistree sous Windows).

Usage : python docs/poster/render_poster.py
"""

from __future__ import annotations

from pathlib import Path

import win32com.client.dynamic

HERE = Path(__file__).resolve().parent
PPTX = HERE / "ICAIS26_Poster_FaceAntiSpoofing.pptx"


def main() -> None:
    app = win32com.client.dynamic.Dispatch("PowerPoint.Application")
    try:
        pres = app.Presentations.Open(str(PPTX), True, False, False)   # lecture seule, sans fenetre
        pres.SaveAs(str(PPTX.with_suffix(".pdf")), 32)                  # 32 = ppSaveAsPDF
        pres.Slides(1).Export(str(PPTX.with_suffix(".png")), "PNG", 2400, 3393)
        pres.Close()
    finally:
        app.Quit()
    print("PDF :", PPTX.with_suffix(".pdf"))
    print("PNG :", PPTX.with_suffix(".png"))


if __name__ == "__main__":
    main()
