import os
import cv2
from ultralytics import YOLO
import uuid

class PlateDetectionService:
    def __init__(self, model_path='models/legacy_plate_detection_models/Legacy_plate.pt', output_dir='outputs'):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image(self, image_path: str) -> dict:
        """
        Detect license plates (no OCR).
        Args:
            image_path (str): Path to the input image.
        Returns:
            dict: Summary with list of detected plates and their info.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        results = self.model(image_path)

        # Save the annotated image directly in outputs/
        annotated_filename = f"legacyplate_{uuid.uuid4().hex}.jpg"
        dest_path = os.path.join(self.output_dir, annotated_filename)
        results[0].save(filename=dest_path)

        output_images = [dest_path]

        # Detection summary for API
        img = cv2.imread(image_path)
        h_img, w_img = img.shape[:2]
        boxes = results[0].boxes

        detected_plates = []
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy.cpu().numpy()[0].astype(int)
            conf = float(box.conf.cpu().numpy()[0])
            class_idx = int(box.cls.cpu().numpy()[0])
            class_label = self.model.names.get(class_idx, str(class_idx))
            if isinstance(class_label, str) and class_label.lower() == "sri":
                class_label = "Sri"
            plate_info = {
                "confidence": conf,
                "bbox": xyxy.tolist(),
                "class": class_label
            }
            detected_plates.append(plate_info)

        return {
            "input_image": image_path,
            "annotated_images": output_images,
            "plates_found": detected_plates
        }


# Example usage
if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Detect license plates in a single JPG image (NO OCR).')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLOv8 weights.')
    parser.add_argument('--img', type=str, required=True, help='Path to input JPG image.')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Directory to save the results.')
    args = parser.parse_args()

    service = PlateDetectionService(model_path=args.model, output_dir=args.save_dir)
    results = service.process_image(args.img)
    print(json.dumps(results, indent=2))
