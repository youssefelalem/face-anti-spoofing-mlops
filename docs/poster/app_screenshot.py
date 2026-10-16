"""Capture d'ecran de l'application pour le poster (navigateur Edge/Chrome sans interface).

L'application doit tourner (uvicorn). Une image de test webcam (session s2, jamais vue a
l'entrainement) est copiee temporairement dans app/static/ puis ouverte via le lien de
demonstration /#src=/static/<fichier>.

Usage : python docs/poster/app_screenshot.py [--url http://localhost:8001] [--attack screen|real]
Sortie : reports/app_screenshot.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
DEMO = ROOT / "app" / "static" / "_demo_frame.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--attack", default="screen")
    ap.add_argument("--index", type=int, default=40)
    args = ap.parse_args()

    files = sorted((config.WEBCAM_DIR / "s2" / args.attack).glob("*.jpg"))
    face = cv2.resize(cv2.imread(str(files[min(args.index, len(files) - 1)])), (300, 300))
    # Cadre 4:3 avec marge neutre, pour que le detecteur retrouve le visage comme en direct
    frame = cv2.copyMakeBorder(face, 90, 90, 170, 170, cv2.BORDER_CONSTANT, value=(46, 40, 34))
    cv2.imwrite(str(DEMO), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = config.REPORTS_DIR / "app_screenshot.png"
    browser = next(b for b in BROWSERS if Path(b).is_file())
    try:
        with tempfile.TemporaryDirectory() as profile:
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--user-data-dir={profile}", "--window-size=1440,960",
                            "--force-device-scale-factor=1.5", "--virtual-time-budget=10000",
                            f"--screenshot={out}", f"{args.url}/#src=/static/{DEMO.name}"],
                           check=True, timeout=120)
    finally:
        DEMO.unlink(missing_ok=True)          # l'image ne reste pas dans le code de l'application
    print("capture :", out)


if __name__ == "__main__":
    main()
