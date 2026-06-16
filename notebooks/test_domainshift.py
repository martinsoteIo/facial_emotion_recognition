from ultralytics import YOLO

models = {
    "yolo8n_640":   "../datasets/combined_dataset/Entrenamientos_YOLO/YOLO8n_640/weights/best.pt",
    "yolo26n_640":  "../datasets/combined_dataset/Entrenamientos_YOLO/YOLO26n_640/weights/best.pt",
    "yolo26n_1280": "../datasets/combined_dataset/Entrenamientos_YOLO/YOLO26n_1280_HighRes_13062026/weights/best.pt",
    "yolo26s_640":  "../datasets/combined_dataset/Entrenamientos_YOLO/YOLO26s_640_Scale3/weights/best.pt",
}

for name, path in models.items():
    model  = YOLO(path)
    result = model.predict("../assets/happy.png", conf=0.10, imgsz=1280, verbose=False)[0]
    print(f"\n{name}:")
    if not result.boxes:
        print("  → Sin detecciones")
    for box in result.boxes:
        lbl  = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"  → {lbl:10s}  conf={conf:.3f}")