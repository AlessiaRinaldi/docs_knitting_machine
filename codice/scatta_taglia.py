#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scatto con Picamera2 + crop automatico (headless).

- Salva gli scatti in /home/aless/foto
- Salva i ritagli in /home/aless/foto/crops
- ROI configurabile in pixel (assoluta) o percentuale (relativa alla risoluzione)

Dipendenze:
  sudo apt update
  sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-numpy
"""

import os, time, glob
from datetime import datetime
from gpiozero import Button
from picamera2 import Picamera2
import cv2
import numpy as np

# ==========================
# CONFIGURAZIONE
# ==========================
SAVE_DIR = "/home/aless/foto"
CROP_DIR = os.path.join(SAVE_DIR, "crops")

# Risoluzione dello scatto (scegli quella che usi normalmente)
CAM_WIDTH, CAM_HEIGHT = 1280, 960   # 1280x960 simile al tuo esempio; puoi mettere 1280x720, 1920x1080, ecc.

# Modalità ROI: "relative" (percentuali 0..1) oppure "pixels" (assoluti)
ROI_MODE = "relative"  # "relative" | "pixels"

# ROI relativa (percentuali 0..1) — esempio: riquadro in basso-centrale come nel tuo mockup
# rx, ry, rw, rh = (x/W, y/H, w/W, h/H)
ROI_REL = (0.48, 0.81, 0.06, 0.10)

# ROI assoluta in pixel (x, y, w, h) — usa questa se preferisci numeri fissi
ROI_PIX = (560, 760, 120, 120)

# Debounce del pulsante (secondi)
BUTTON_BOUNCE = 0.05

# Pin fisico: GPIO17 (pin 11)
BUTTON_PIN = 17

# Processare all'avvio anche le foto già presenti?
PROCESS_EXISTING_ON_START = True

# Formato salvataggio: "jpg" o "png" (jpg più leggero)
SAVE_EXT = "jpg"


# ==========================
# UTILS
# ==========================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

def roi_from_mode(W, H):
    if ROI_MODE == "relative":
        rx, ry, rw, rh = ROI_REL
        x = int(rx * W)
        y = int(ry * H)
        w = int(rw * W)
        h = int(rh * H)
    else:
        x, y, w, h = ROI_PIX
    return clamp_roi(x, y, w, h, W, H)

def crop_and_save(img_path):
    """Ritaglia l'immagine e salva in CROP_DIR con suffisso _crop."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Impossibile leggere {img_path}")
        return None

    H, W = img.shape[:2]
    x, y, w, h = roi_from_mode(W, H)
    crop = img[y:y+h, x:x+w]

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(CROP_DIR, f"{base}_crop.png")  # ritaglio sempre lossless
    ok = cv2.imwrite(out_path, crop)
    if ok:
        print(f"[OK] Ritaglio -> {out_path} (ROI x={x} y={y} w={w} h={h}, img={W}x{H})")
        return out_path
    else:
        print(f"[ERR] Salvataggio ritaglio fallito per {img_path}")
        return None


# ==========================
# INIZIALIZZAZIONE CAMERA / BUTTON
# ==========================
ensure_dir(SAVE_DIR)
ensure_dir(CROP_DIR)

button = Button(BUTTON_PIN, pull_up=True, bounce_time=BUTTON_BOUNCE)

picam2 = Picamera2()
# Config semplice per foto (still)
still_cfg = picam2.create_still_configuration(
    main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "XRGB8888"}
)
picam2.configure(still_cfg)
picam2.start()
time.sleep(0.3)  # piccolo warm-up

# ==========================
# CALLBACK DI SCATTO
# ==========================
def scatta_foto():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(SAVE_DIR, f"shot_{ts}.{SAVE_EXT}")
    try:
        print(f"[INFO] Scatto -> {filepath}")
        # Salva direttamente su file (JPEG/PNG)
        picam2.capture_file(filepath)
        # Subito dopo: crop e salvataggio
        crop_and_save(filepath)
    except Exception as e:
        print(f"[ERR] Errore durante lo scatto: {e}")

button.when_pressed = scatta_foto


# ==========================
# PROCESSA IMMAGINI GIÀ PRESENTI (OPZIONALE)
# ==========================
def process_existing():
    patt = []
    patt += glob.glob(os.path.join(SAVE_DIR, "*.jpg"))
    patt += glob.glob(os.path.join(SAVE_DIR, "*.jpeg"))
    patt += glob.glob(os.path.join(SAVE_DIR, "*.png"))

    # Evita di ricroppare file già processati
    done = set(os.path.splitext(os.path.basename(p))[0].replace("_crop", "")
               for p in glob.glob(os.path.join(CROP_DIR, "*_crop.png")))

    to_do = [p for p in patt if os.path.splitext(os.path.basename(p))[0] not in done]

    if not to_do:
        print("[INFO] Nessuna immagine da processare in backlog.")
        return
    print(f"[INFO] Processazione backlog: {len(to_do)} file...")
    for p in sorted(to_do):
        crop_and_save(p)

if PROCESS_EXISTING_ON_START:
    process_existing()

# ==========================
# MAIN LOOP
# ==========================
print("[OK] Pronto. Premi il microinterruttore per scattare (CTRL+C per uscire).")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    print("\n[OK] Uscita.")
