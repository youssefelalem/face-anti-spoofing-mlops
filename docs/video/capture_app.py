"""Captures de l'application avec le visage de l'auteur (session s3), pour la video et le README.

L'application doit tourner. Pour chaque type (real, screen), l'image est copiee
temporairement dans app/static/ et analysee via le lien /#src=..., puis la zone
video + verdict est recadree.

Usage : python docs/video/capture_app.py [--url http://localhost:8501]
Sorties : docs/video/app_real.png, docs/video/app_attack.png, docs/assets/app_demo.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
HERE = Path(__file__).resolve().parent
DEMO = ROOT / "app" / "static" / "_demo_frame.jpg"


def shoot(url: str, src: Path, out: Path) -> None:
    face = cv2.resize(cv2.imread(str(src)), (300, 300))
    frame = cv2.copyMakeBorder(face, 90, 90, 170, 170, cv2.BORDER_CONSTANT, value=(46, 40, 34))
    cv2.imwrite(str(DEMO), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    browser = next(b for b in BROWSERS if Path(b).is_file())
    try:
        with tempfile.TemporaryDirectory() as profile:
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--user-data-dir={profile}", "--window-size=1440,960",
                            "--force-device-scale-factor=1.5", "--virtual-time-budget=10000",
                            f"--screenshot={out}", f"{url}/#src=/static/{DEMO.name}"],
                           check=True, timeout=120)
    finally:
        DEMO.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--session", default="s3")
    args = ap.parse_args()
    (ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    for kind, name in [("real", "app_real"), ("screen", "app_attack")]:
        files = sorted((config.WEBCAM_DIR / args.session / kind).glob("*.jpg"))
        full = HERE / f"{name}_full.png"
        shoot(args.url, files[len(files) // 2], full)
        im = Image.open(full).convert("RGB")
        # Zone video et carte verdict (pixels de la capture 2160x1440), empilees verticalement
        view = im.crop((244, 257, 1376, 1106))
        verdict = im.crop((1434, 154, 1944, 461))
        verdict = verdict.resize((view.width, int(verdict.height * view.width / verdict.width)))
        stack = Image.new("RGB", (view.width, view.height + verdict.height + 24), im.getpixel((5, 400)))
        stack.paste(view, (0, 0))
        stack.paste(verdict, (0, view.height + 24))
        stack.save(HERE / f"{name}.png")
        if kind == "real":
            im.save(ROOT / "docs" / "assets" / "app_demo.png")          # README
        else:
            im.save(ROOT / "reports" / "app_screenshot.png")            # poster
        full.unlink()
        print("capture :", HERE / f"{name}.png")


if __name__ == "__main__":
    main()
