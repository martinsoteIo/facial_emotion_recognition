"""
benchmark_models.py
───────────────────
Prueba los 4 modelos YOLO sobre un fragmento corto del vídeo (por defecto 30 s)
y genera un informe comparativo: vídeos anotados + gráficas + resumen CSV.

Uso:
    python benchmark_models.py
    python benchmark_models.py --start 60 --duration 45 --conf 0.25
    python benchmark_models.py --conf 0.20 --imgsz_override 1280

Ctrl+C en cualquier momento: guarda lo procesado hasta ese punto.
"""

import os
import cv2
import csv
import signal
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO
from tqdm import tqdm

# ── Ctrl+C limpio ────────────────────────────────────────────────────────────
_stop = False
def _sig(s, f):
    global _stop
    print("\n[WARN] Interrupción recibida — cerrando limpiamente…")
    _stop = True
signal.signal(signal.SIGINT, _sig)


# ── Definición de los 4 modelos ───────────────────────────────────────────────
BASE = "../datasets/combined_dataset/Entrenamientos_YOLO"

MODELS = [
    {
        "id":    "yolo8n_640",
        "label": "YOLOv8n  imgsz=640",
        "path":  f"{BASE}/YOLO8n_640/weights/best.pt",
        "imgsz": 640,
    },
    {
        "id":    "yolo26n_640",
        "label": "YOLOv26n imgsz=640",
        "path":  f"{BASE}/YOLO26n_640/weights/best.pt",
        "imgsz": 640,
    },
    {
        "id":    "yolo26n_1280",
        "label": "YOLOv26n imgsz=1280",
        "path":  f"{BASE}/YOLO26n_1280_HighRes_14062026/weights/best.pt",
        "imgsz": 1280,
    },
    {
        "id":    "yolo26s_640",
        "label": "YOLOv26s imgsz=640",
        "path":  f"{BASE}/YOLO26s_640_Scale3/weights/best.pt",
        "imgsz": 640,
    },
]


# ── Colores por emoción (BGR) ─────────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":    (0,   200, 80),
    "sad":      (200,  60,  0),
    "angry":    (0,    0, 220),
    "surprise": (0,   180, 255),
    "fear":     (130,  0, 200),
    "disgust":  (0,   140, 100),
    "neutral":  (160, 160, 160),
}
DEFAULT_COLOR = (200, 200, 200)


def get_color(emotion: str):
    return EMOTION_COLORS.get(emotion.lower(), DEFAULT_COLOR)


def extract_fragment(video_path: str, start_sec: float, duration_sec: float):
    """Devuelve (frames_list, fps, width, height)."""
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = int(start_sec * fps)
    total = int(duration_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(total):
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(frm)
    cap.release()
    return frames, fps, w, h


def run_model(cfg: dict, frames: list, conf: float,
              imgsz_override: int | None, device: str, out_dir: str):
    """
    Ejecuta el modelo sobre todos los frames.
    Devuelve dict con métricas y escribe el vídeo anotado.
    """
    imgsz = imgsz_override if imgsz_override else cfg["imgsz"]

    if not os.path.exists(cfg["path"]):
        print(f"  [SKIP] Pesos no encontrados: {cfg['path']}")
        return None

    model       = YOLO(cfg["path"])
    class_names = list(model.names.values())

    # Salida de vídeo
    fps_out    = 25
    h, w       = frames[0].shape[:2]
    vid_path   = os.path.join(out_dir, f"{cfg['id']}_annotated.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(vid_path, fourcc, fps_out, (w, h))

    timeline   = {e: [] for e in class_names}
    total_dets = 0
    fps_times  = []

    print(f"\n  ▶  {cfg['label']}  |  imgsz={imgsz}  |  {len(frames)} frames")

    with tqdm(frames, desc=f"    {cfg['id']}", unit="frame", colour="cyan",
              leave=False) as bar:
        for frm in bar:
            if _stop:
                break
            t0     = time.perf_counter()
            result = model.predict(
                source=frm, conf=conf, imgsz=imgsz,
                device=device, verbose=False, save=False,
            )[0]
            fps_times.append(time.perf_counter() - t0)

            counts = {e: 0 for e in class_names}
            for box in result.boxes:
                lbl = model.names[int(box.cls[0])]
                counts[lbl] += 1
                total_dets  += 1

            for e in class_names:
                timeline[e].append(counts[e])

            # Dibujar anotación manual con colores de emoción
            ann = frm.copy()
            img_h, img_w = ann.shape[:2]
            for box in result.boxes:
                lbl      = model.names[int(box.cls[0])]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = get_color(lbl)

                # Caja de detección
                cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)

                tag         = f"{lbl} {conf_val:.2f}"
                tag_w       = len(tag) * 9
                tag_h       = 22
                margin      = 4   # píxeles mínimos desde el borde

                # Si hay espacio arriba → etiqueta encima; si no → debajo
                if y1 - tag_h - margin >= 0:
                    rect_y1 = y1 - tag_h
                    rect_y2 = y1
                    text_y  = y1 - 5
                else:
                    rect_y1 = y2
                    rect_y2 = y2 + tag_h
                    text_y  = y2 + tag_h - 5

                # Recortar para que no salga por la derecha ni abajo
                rect_x2 = min(x1 + tag_w, img_w)
                rect_y2 = min(rect_y2, img_h)

                cv2.rectangle(ann, (x1, rect_y1), (rect_x2, rect_y2), color, -1)
                cv2.putText(ann, tag, (x1 + 2, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            writer.write(ann)
            bar.set_postfix_str(
                f"dets={sum(counts.values())} "
                f"fps={1/fps_times[-1]:.1f}"
            )

    writer.release()

    avg_fps      = 1 / np.mean(fps_times) if fps_times else 0
    frames_w_det = sum(1 for e in class_names
                       for c in timeline[e] if c > 0)
    coverage_pct = 100 * frames_w_det / max(len(frames), 1)

    return {
        "id":           cfg["id"],
        "label":        cfg["label"],
        "imgsz":        imgsz,
        "total_dets":   total_dets,
        "avg_fps":      round(avg_fps, 2),
        "coverage_pct": round(coverage_pct, 1),
        "timeline":     timeline,
        "class_names":  class_names,
        "vid_path":     vid_path,
    }


def plot_comparison(results: list, out_dir: str, duration_sec: float):
    """Genera figura comparativa 2×2 con la timeline de cada modelo."""
    valid = [r for r in results if r is not None]
    if not valid:
        return

    n    = len(valid)
    cols = 2
    rows = (n + 1) // cols

    fig = plt.figure(figsize=(14, 4 * rows), dpi=180)
    fig.suptitle("Comparativa de Modelos — Detecciones por Emoción",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35)

    palette = plt.cm.tab10.colors

    for i, r in enumerate(valid):
        ax  = fig.add_subplot(gs[i // cols, i % cols])
        fps = max(duration_sec / max(len(list(r["timeline"].values())[0]), 1), 0.001)
        t   = np.arange(len(list(r["timeline"].values())[0])) * fps

        for j, e in enumerate(r["class_names"]):
            counts   = np.array(r["timeline"][e])
            smoothed = np.convolve(counts, np.ones(5) / 5, mode="same")
            if counts.sum() > 0:
                ax.plot(t, smoothed, label=e.capitalize(),
                        color=palette[j % len(palette)], linewidth=1.8)

        ax.set_title(
            f"{r['label']}\n"
            f"dets={r['total_dets']}  cov={r['coverage_pct']}%  "
            f"{r['avg_fps']:.1f} fps",
            fontsize=9, pad=6
        )
        ax.set_xlabel("Segundos", fontsize=8)
        ax.set_ylabel("Caras detectadas", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_comparison.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfica comparativa guardada: {path}")


def save_summary_csv(results: list, out_dir: str):
    """Guarda un CSV con las métricas clave de cada modelo."""
    path = os.path.join(out_dir, "benchmark_summary.csv")
    fields = ["model", "label", "imgsz", "total_detections",
              "coverage_pct", "avg_fps_cpu"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            if r is None:
                continue
            w.writerow({
                "model":            r["id"],
                "label":            r["label"],
                "imgsz":            r["imgsz"],
                "total_detections": r["total_dets"],
                "coverage_pct":     r["coverage_pct"],
                "avg_fps_cpu":      r["avg_fps"],
            })
    print(f"[INFO] Resumen CSV guardado: {path}")


def make_grid_video(results: list, frames_orig: list, out_dir: str, fps: float):
    """
    Vídeo 2×2 con los 4 modelos corriendo en paralelo frame a frame.
    Cada cuadrante lleva el nombre del modelo y un contador de detecciones.
    El vídeo original sin anotar va en la esquina superior-izquierda si
    hay menos de 4 resultados válidos.
    """
    valid = [r for r in results if r is not None]
    if not valid:
        return

    # Tamaño de cada celda (reducimos a la mitad para que entre en pantalla)
    h_orig, w_orig = frames_orig[0].shape[:2]
    cell_w = w_orig // 2
    cell_h = h_orig // 2
    grid_w = cell_w * 2
    grid_h = cell_h * 2

    out_path = os.path.join(out_dir, "benchmark_grid_2x2.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (grid_w, grid_h))

    # Re-ejecutar inferencia frame a frame para tener los frames anotados en memoria
    # (evitamos re-leer los mp4 ya escritos, que pueden tener codec distinto)
    print("\n[INFO] Generando vídeo comparativo 2×2…")

    # Cargar modelos una vez
    loaded = []
    for r in results:
        if r is None:
            loaded.append(None)
            continue
        cfg_match = next((c for c in MODELS if c["id"] == r["id"]), None)
        if cfg_match and os.path.exists(cfg_match["path"]):
            m = YOLO(cfg_match["path"])
            loaded.append((r, m, cfg_match["imgsz"]))
        else:
            loaded.append(None)

    # Rellenar hasta 4 slots con None
    while len(loaded) < 4:
        loaded.append(None)

    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]

    with tqdm(frames_orig, desc="  Grid 2×2", unit="frame", colour="yellow") as bar:
        for frm in bar:
            if _stop:
                break

            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for slot_idx, item in enumerate(loaded[:4]):
                px, py = positions[slot_idx]

                if item is None:
                    # Slot vacío: frame original redimensionado
                    cell = cv2.resize(frm, (cell_w, cell_h))
                    cv2.putText(cell, "No disponible", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                    grid[py:py+cell_h, px:px+cell_w] = cell
                    continue

                r, model, imgsz = item
                result = model.predict(
                    source=frm, conf=args_global["conf"],
                    imgsz=imgsz, device=args_global["device"],
                    verbose=False, save=False,
                )[0]

                # Dibujar cajas con colores de emoción
                ann = frm.copy()
                img_h, img_w = ann.shape[:2]
                det_count = 0
                for box in result.boxes:
                    lbl      = model.names[int(box.cls[0])]
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = get_color(lbl)
                    cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)

                    tag    = f"{lbl} {conf_val:.2f}"
                    tag_w  = len(tag) * 10
                    tag_h  = 26
                    margin = 4

                    if y1 - tag_h - margin >= 0:
                        rect_y1 = y1 - tag_h
                        rect_y2 = y1
                        text_y  = y1 - 6
                    else:
                        rect_y1 = y2
                        rect_y2 = y2 + tag_h
                        text_y  = y2 + tag_h - 6

                    rect_x2 = min(x1 + tag_w, img_w)
                    rect_y2 = min(rect_y2, img_h)

                    cv2.rectangle(ann, (x1, rect_y1), (rect_x2, rect_y2), color, -1)
                    cv2.putText(ann, tag, (x1 + 2, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    det_count += 1

                # Reducir a tamaño de celda
                cell = cv2.resize(ann, (cell_w, cell_h))

                # Cabecera semitransparente con nombre del modelo
                overlay = cell.copy()
                cv2.rectangle(overlay, (0, 0), (cell_w, 36), (0, 0, 0), -1)
                cell = cv2.addWeighted(overlay, 0.55, cell, 0.45, 0)
                short_label = r["label"]
                cv2.putText(cell, short_label, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
                # Contador de detecciones (esquina sup. derecha)
                det_txt = f"dets: {det_count}"
                tw = cv2.getTextSize(det_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                cv2.putText(cell, det_txt, (cell_w - tw - 8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 1)

                grid[py:py+cell_h, px:px+cell_w] = cell

            # Líneas divisorias
            cv2.line(grid, (cell_w, 0), (cell_w, grid_h), (60, 60, 60), 2)
            cv2.line(grid, (0, cell_h), (grid_w, cell_h), (60, 60, 60), 2)

            writer.write(grid)

    writer.release()
    print(f"[INFO] Vídeo comparativo guardado: {out_path}")
    return out_path


def print_ranking(results: list):
    valid = [r for r in results if r is not None]
    if not valid:
        return
    ranked = sorted(valid, key=lambda r: (r["total_dets"], r["coverage_pct"]),
                    reverse=True)
    print("\n" + "═" * 60)
    print("  RANKING  (mayor detecciones → mejor)")
    print("═" * 60)
    medals = ["🥇", "🥈", "🥉", "  "]
    for i, r in enumerate(ranked):
        m = medals[min(i, 3)]
        print(f"  {m}  {r['label']:<26} "
              f"dets={r['total_dets']:>4}  "
              f"cov={r['coverage_pct']:>5.1f}%  "
              f"{r['avg_fps']:>5.1f} fps")
    print("═" * 60)
    print(f"\n  ✔  Modelo recomendado: {ranked[0]['label']}")
    print(f"     Vídeo anotado en  : {ranked[0]['vid_path']}\n")


# Dict global para que make_grid_video acceda a conf y device sin pasar args
args_global = {}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",          type=str,   default="../assets/camera_D_0_06.mp4")
    parser.add_argument("--start",          type=float, default=0,
                        help="Segundo de inicio del fragmento (default: 30)")
    parser.add_argument("--duration",       type=float, default=30,
                        help="Duración del fragmento en segundos (default: 30)")
    parser.add_argument("--conf",           type=float, default=0.25)
    parser.add_argument("--output_dir",     type=str,   default="../runs/benchmark_05")
    parser.add_argument("--imgsz_override", type=int,   default=None,
                        help="Forzar imgsz para TODOS los modelos (útil para debug)")
    parser.add_argument("--no_grid",        action="store_true",
                        help="Saltar la generación del vídeo 2×2 (más rápido)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Rellenar dict global
    args_global["conf"]   = args.conf
    args_global["device"] = device

    print("\n" + "═" * 60)
    print("  BENCHMARK COMPARATIVO — 4 MODELOS YOLO")
    print("═" * 60)
    print(f"  Vídeo   : {args.video}")
    print(f"  Segmento: {args.start}s → {args.start + args.duration}s  "
          f"({args.duration}s)")
    print(f"  Hardware: {device.upper()}")
    print(f"  Conf    : {args.conf}")
    print(f"  Salida  : {out_dir}")
    print("═" * 60)

    # 1. Extraer fragmento una sola vez (no re-leer el vídeo 4 veces)
    print(f"\n[INFO] Extrayendo fragmento del vídeo ({args.duration}s)…")
    frames, fps, w, h = extract_fragment(args.video, args.start, args.duration)
    print(f"[INFO] {len(frames)} frames extraídos  ({w}×{h} @ {fps:.1f} fps)")

    if not frames:
        print("[ERROR] No se pudieron extraer frames. Revisa la ruta del vídeo.")
        return

    # 2. Ejecutar cada modelo
    results = []
    for cfg in MODELS:
        if _stop:
            break
        r = run_model(cfg, frames, args.conf, args.imgsz_override, device, out_dir)
        results.append(r)

    # 3. Informe
    plot_comparison(results, out_dir, args.duration)
    save_summary_csv(results, out_dir)
    print_ranking(results)

    # 4. Vídeo comparativo 2×2
    if not args.no_grid and not _stop:
        make_grid_video(results, frames, out_dir, fps)

    print("\n[INFO] Todo listo. Archivos en:", out_dir)
    print("       Abre benchmark_grid_2x2.mp4 para ver los 4 modelos en paralelo.")


if __name__ == "__main__":
    main()
