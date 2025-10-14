#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riconoscimento headless 'U' corretta vs 'riga orizzontale' su crop di uncini.

Dipendenze:
  sudo apt update
  sudo apt install -y python3-opencv python3-numpy
Esecuzione tipica:
  python3 detect_knit_errors.py --in-dir /home/aless/foto/crops --csv /home/aless/foto/results.csv
"""

import os, glob, csv, argparse, sys
import numpy as np
import cv2
from datetime import datetime

def normalize_img(bgr, target=(96, 96)):
    """Grayscale + CLAHE (robusto a luci/colore del filo) + resize standard."""
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr
    # CLAHE: migliora contrasto locale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    if target:
        g = cv2.resize(g, target, interpolation=cv2.INTER_AREA)
    return g

def grad_ratio_and_lines(g):
    """Calcola rapporto di gradiente e linee orizzontali rilevate."""
    # Scharr = più robusto del Sobel su dettagli fini
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    Sx = float(np.sum(np.abs(gx)))  # vertical edges (cambio lungo x)
    Sy = float(np.sum(np.abs(gy)))  # horizontal edges (cambio lungo y)
    ratio = Sx / (Sy + 1e-6)

    # Canny + Hough per confermare linee quasi orizzontali
    edges = cv2.Canny(g, 30, 90, L2gradient=True)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20,
                            minLineLength=int(0.45 * g.shape[1]), maxLineGap=8)
    has_horiz = False
    longest = 0
    if lines is not None:
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length > longest:
                longest = length
            # orizzontale se |angolo| < 12°
            if abs(dy) <= abs(dx) * np.tan(np.deg2rad(12)):
                has_horiz = True
    return ratio, has_horiz, longest

def classify_image(gray_norm,
                   ratio_thresh=0.85,   # < 0.85 => probabile riga orizzontale
                   require_hough=True):
    ratio, has_h, longest = grad_ratio_and_lines(gray_norm)
    if ratio < ratio_thresh and (has_h or not require_hough):
        return "ERROR_LINE", ratio, has_h, longest
    else:
        return "OK_U", ratio, has_h, longest

def process_dir(in_dir, csv_path=None, verbose=True):
    paths = []
    exts = ("*.png","*.jpg","*.jpeg","*.bmp")
    for e in exts:
        paths += glob.glob(os.path.join(in_dir, e))
    paths.sort()
    if not paths:
        print(f"[ERR] Nessuna immagine trovata in: {in_dir}")
        sys.exit(1)

    writer = None
    fcsv = None
    if csv_path:
        fcsv = open(csv_path, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["timestamp","file","class","ratio_Sx_Sy","hough_horizontal","longest_px"])

    ok, err = 0, 0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Impossibile leggere {p}")
            continue
        g = normalize_img(img)  # 96x96
        clas, ratio, has_h, longest = classify_image(g)
        if clas == "OK_U":
            ok += 1
        else:
            err += 1
        if verbose:
            print(f"{os.path.basename(p)} -> {clas} | ratio={ratio:.2f} | horiz_line={has_h} | longest={int(longest)}px")
        if writer:
            writer.writerow([datetime.utcnow().isoformat()+"Z", os.path.basename(p), clas,
                             f"{ratio:.4f}", int(has_h), int(longest)])
    if fcsv:
        fcsv.close()
    print(f"\n[SUMMARY] OK_U: {ok} | ERROR_LINE: {err} | tot: {ok+err}")

def main():
    ap = argparse.ArgumentParser(description="Riconoscimento U vs riga orizzontale (headless).")
    ap.add_argument("--in-dir", required=True, help="Cartella con i crop (es. /home/aless/foto/crops)")
    ap.add_argument("--csv", default="", help="CSV di output (opzionale)")
    ap.add_argument("--quiet", action="store_true", help="Meno stampe a schermo")
    args = ap.parse_args()
    process_dir(args.in_dir, csv_path=args.csv or None, verbose=not args.quiet)

if __name__ == "__main__":
    main()
