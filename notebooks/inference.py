import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Mood Inference Script")
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to the input image for mood deduction"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="../datasets/combined_dataset/Entrenamientos_YOLO/YOLO26s_640_Scale3/weights/best.pt", 
        help="Path to the trained YOLO weights (.pt file)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../runs/detect/", 
        help="Root directory to save annotated results"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold \u03c4 for filtering predictions"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found at: {args.image}")
        return
    
    abs_output_dir = os.path.abspath(args.output_dir)

    print(f"[INFO] Loading mood detection model from: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Running inference on: {args.image}")
    results = model.predict(
        source=args.image,
        conf=args.conf,
        device="cpu",
        imgsz=1280,
        save=True,
        project=abs_output_dir,
        name="predicciones_mood",
        exist_ok=True
    )

    print("\n" + "="*50)
    print("               DETECTION RESULTS                ")
    print("="*50)
    
    detected_anything = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detected_anything = True
            # Get class ID, confidence score, and map to label name
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            mood_label = model.names[class_id]
            
            # Extract bounding box coordinates: [x_min, y_min, x_max, y_max]
            bbox = box.xyxy[0].tolist()
            bbox_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"

            print(f"• Detected Mood: {mood_label.upper()}")
            print(f"  Confidence:    {confidence * 100:.2f}%")
            print(f"  Bounding Box:  {bbox_str}\n")

    if not detected_anything:
        print("[INFO] No mood could be confidently deduced with the current threshold \u03c4.")
    
    print("="*50)
    print(f"[INFO] Annotated image saved to: {args.output_dir}/predicciones_mood/")

if __name__ == "__main__":
    main()

'''
import os
import cv2
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Advanced YOLO Mood Inference Script with CLAHE")
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to the input image for mood deduction"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="../datasets/combined_dataset/Entrenamientos_YOLO/YOLO26s_640_Scale3/weights/best.pt", 
        help="Path to the trained YOLO weights (.pt file)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../runs", 
        help="Root directory to save annotated results"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold \u03c4 for filtering predictions"
    )
    parser.add_argument(
        "--preprocess",
        type=bool,
        default=True,
        help="Apply CLAHE local contrast enhancement to break neutral bias"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found at: {args.image}")
        return
    
    # Resolver la ruta absoluta exacta para evitar duplicaciones de YOLO
    abs_output_dir = os.path.abspath(args.output_dir)

    # 1. Cargar la imagen original mediante OpenCV
    img = cv2.imread(args.image)
    source_input = args.image

    # 2. CAPA DE MEJORA ÓPTICA: Si el preprocesamiento está activo, optimizamos la matriz
    if args.preprocess:
        print("[INFO] Aplicando filtro CLAHE para corregir oclusiones lumínicas y sombras...")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Filtro adaptativo local para resaltar microexpresiones tapadas por las gafas
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Recomponer imagen optimizada
        merged_lab = cv2.merge((cl, a_channel, b_channel))
        img_enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        
        # Guardamos un archivo temporal para que YOLO conserve el nombre original al salvar las cajas
        base_name = os.path.basename(args.image)
        source_input = os.path.join(os.path.dirname(args.image), f"enhanced_{base_name}")
        cv2.imwrite(source_input, img_enhanced)

    print(f"[INFO] Loading mood detection model from: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Running inference at high resolution (1280x1280) on: {source_input}")
    results = model.predict(
        source=source_input,
        conf=args.conf,
        device="cpu",
        imgsz=1280,        # Mantiene la alta resolución para rostros pequeños
        save=True,
        project=abs_output_dir,
        name="predicciones_mood",
        exist_ok=True
    )

    print("\n" + "="*50)
    print("               DETECTION RESULTS                ")
    print("="*50)
    
    detected_anything = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detected_anything = True
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            mood_label = model.names[class_id]
            
            bbox = box.xyxy[0].tolist()
            bbox_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"

            print(f"• Detected Mood: {mood_label.upper()}")
            print(f"  Confidence:    {confidence * 100:.2f}%")
            print(f"  Bounding Box:  {bbox_str}\n")

    if not detected_anything:
        print("[INFO] No mood could be confidently deduced with the current threshold \u03c4.")
    
    print("="*50)
    
    # Limpieza del archivo temporal si se creó
    if args.preprocess and os.path.exists(source_input):
        os.remove(source_input)

    final_save_path = os.path.join(abs_output_dir, "predicciones_mood")
    print(f"[INFO] Processed image saved to code-ready path: {final_save_path}")

if __name__ == "__main__":
    main()
'''