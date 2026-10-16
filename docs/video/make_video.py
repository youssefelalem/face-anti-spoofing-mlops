"""Video LinkedIn (1080x1080, ~50 s, sans son) a partir des figures et captures du projet.

Prerequis : figures (docs/poster/make_figures.py, src/fusion.py), captures de l'application
avec le visage de l'auteur (docs/video/capture_app.py), ffmpeg dans le PATH.
Aucune image NUAA ni visage d'un tiers.

Usage : python docs/video/make_video.py
Sortie : docs/video/face_antispoofing_linkedin.mp4
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
OUT = HERE / "face_antispoofing_linkedin.mp4"
W = H = 1080
FPS = 30
FADE = 0.5

NAVY, WHITE, ICE, ORANGE = (11, 30, 63), (255, 255, 255), (232, 238, 248), (217, 130, 43)
GREEN, RED, MUTED = (29, 138, 78), (196, 50, 43), (92, 101, 118)
FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"

FINAL = json.loads((ROOT / "reports" / "final_metrics.json").read_text(encoding="utf-8"))["test"]
V1 = json.loads((ROOT / "reports" / "test_metrics_v1_subject_protocol.json").read_text(encoding="utf-8"))
DQ = {c["check"]: c for c in json.loads((ROOT / "reports" / "data_quality_report.json")
                                         .read_text(encoding="utf-8"))["checks"]}
APP = json.loads((ROOT / "reports" / "app_benchmark.json").read_text(encoding="utf-8"))


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def text_block(draw, xy, lines, size, color, bold=False, spacing=1.25, anchor_center=False):
    x, y = xy
    f = font(size, bold)
    for line in lines:
        if anchor_center:
            w = draw.textlength(line, font=f)
            draw.text(((W - w) / 2, y), line, font=f, fill=color)
        else:
            draw.text((x, y), line, font=f, fill=color)
        y += int(size * spacing)
    return y


def card(bg, title=None, kicker=None):
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    y = 80
    dark = sum(bg) < 300
    if kicker:
        d.text((80, y), kicker.upper(), font=font(30, True), fill=ORANGE)
        y += 52
    if title:
        y = text_block(d, (80, y), title, 58, WHITE if dark else NAVY, bold=True, spacing=1.18)
    return im, d, y + 20


def fit(img: Image.Image, box_w, box_h) -> Image.Image:
    s = min(box_w / img.width, box_h / img.height)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


def figure_scene(path, kicker, title, caption):
    im, d, y = card(WHITE, title, kicker)
    fig = fit(Image.open(path).convert("RGB"), W - 120, H - y - 170)
    im.paste(fig, ((W - fig.width) // 2, y + 10))
    text_block(d, (80, H - 140), caption, 30, MUTED)
    return im


def pct(v):
    return f"{v * 100:.1f} %"


def scenes() -> list[tuple[Image.Image, float]]:
    out = []
    # 1. Titre
    im, d, _ = card(NAVY)
    text_block(d, (0, 300), ["Can a photo on a phone", "fool face recognition?"], 70, WHITE, True,
               anchor_center=True)
    text_block(d, (0, 520), ["Face Anti-Spoofing  ·  Deep Learning + MLOps"], 38, ORANGE, True,
               anchor_center=True)
    text_block(d, (0, 900), ["Youssef EL ALEM  ·  Master DSBD  ·  ICAIS'26"], 30, ICE, anchor_center=True)
    out.append((im, 4.0))

    # 2. Demo (captures avec le visage de l'auteur)
    real, attack = HERE / "app_real.png", HERE / "app_attack.png"
    if real.is_file() and attack.is_file():
        im, d, y = card(WHITE, ["Live face or attack?"], "Real-time demo")
        for i, (p, label, col) in enumerate([(real, "LIVE FACE", GREEN), (attack, "PHONE SCREEN", RED)]):
            shot = fit(Image.open(p).convert("RGB"), 450, 600)
            x = 70 + i * 490
            im.paste(shot, (x, y + 40))
            d.text((x, y + 50 + shot.height), label, font=font(34, True), fill=col)
        caption = f"One webcam frame, ~{APP['latency_ms_median']:.0f} ms on a laptop CPU, no GPU."
        text_block(d, (80, H - 120), [caption], 30, MUTED)
        out.append((im, 5.0))

    # 3. Architecture
    out.append((figure_scene(FIG / "fusion_architecture.png", "Method", ["Two detectors, one decision"],
                             ["A MobileNetV2 CNN (transfer learning) and an LBP texture model,",
                              "calibrated and fused."]), 5.5))
    # 4. Constat principal
    out.append((figure_scene(FIG / "validation_gap.png", "Finding 1", ["Validation can be misleading"],
                             [f"CNN: AUC {V1['validation']['auc']:.4f} on validation,",
                              f"{V1['auc']:.2f} on a new recording session."]), 6.0))
    # 5. Data contract
    lum = DQ["biais_luminance_moyenne"]["detail"]["auc_feature_seule"]
    sat = DQ["biais_saturation_moyenne"]["detail"]["auc_feature_seule"]
    im, d, y = card(NAVY, ["The data contract", "caught the shortcut"], "Finding 2")
    y = text_block(d, (80, y + 40), ["In the training data, one trivial", "feature separated the classes:"],
                   40, ICE, spacing=1.3)
    for label, v in [("Brightness alone", lum), ("Colour saturation alone", sat)]:
        d.text((80, y + 30), label, font=font(40), fill=WHITE)
        d.text((W - 80 - d.textlength(f"AUC {v:.2f}", font=font(56, True)), y + 18), f"AUC {v:.2f}",
               font=font(56, True), fill=ORANGE)
        y += 110
    text_block(d, (80, y + 50), ["The CNN learned it, and failed when", "the lighting changed."], 40, ICE,
               spacing=1.3)
    out.append((im, 6.0))
    # 6. Resultats
    w = FINAL.get("webcam") or FINAL["nuaa"]
    out.append((figure_scene(FIG / "eer_comparison.png", "Finding 3", ["Texture makes the CNN robust"],
                             ["New person, new webcam session: EER "
                              f"{pct(w['CNN (MobileNetV2)']['eer'])} (CNN)",
                              f"-> {pct(w['Fusion']['eer'])} with fusion."]), 6.0))
    # 7. MLOps
    out.append((figure_scene(FIG / "mlops_pipeline.png", "Engineering", ["A reproducible MLOps pipeline"],
                             [f"{APP['n_quality_checks']} data checks · {APP['n_tests']} automated tests · "
                              "MLflow · CI · ONNX · Docker"]), 5.5))
    # 8. Fin
    im, d, _ = card(NAVY)
    text_block(d, (0, 330), ["Face Anti-Spoofing"], 72, WHITE, True, anchor_center=True)
    text_block(d, (0, 440), ["CNN + texture fusion  ·  live demo  ·  MLOps"], 36, ORANGE, True,
               anchor_center=True)
    text_block(d, (0, 620), ["Youssef EL ALEM", "Master Data Science & Big Data", "ICAIS'26 · Casablanca"],
               38, ICE, anchor_center=True, spacing=1.4)
    text_block(d, (0, 900), ["Code on GitHub: github.com/youssefelalem"], 30, ICE, anchor_center=True)
    out.append((im, 4.5))
    return out


def main() -> None:
    items = scenes()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-preset", "slow", "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    arrays = [np.asarray(im, dtype=np.float32) for im, _ in items]
    for i, ((_, dur), a) in enumerate(zip(items, arrays, strict=True)):
        n = int(dur * FPS)
        nxt = arrays[i + 1] if i + 1 < len(arrays) else None
        for f in range(n):
            t = f / FPS
            # leger zoom avant (effet "Ken Burns") pour donner du mouvement
            z = 1 + 0.03 * f / n
            frame = Image.fromarray(a.astype(np.uint8)).resize((int(W * z), int(H * z)), Image.BILINEAR)
            ox, oy = (frame.width - W) // 2, (frame.height - H) // 2
            cur = np.asarray(frame.crop((ox, oy, ox + W, oy + H)), dtype=np.float32)
            if nxt is not None and t > dur - FADE:
                alpha = (t - (dur - FADE)) / FADE
                cur = cur * (1 - alpha) + nxt * alpha
            if i == 0 and t < FADE:
                cur = cur * (t / FADE)                              # fondu d'entree
            proc.stdin.write(cur.clip(0, 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("Video :", OUT, f"({sum(d for _, d in items):.0f} s)")


if __name__ == "__main__":
    main()
