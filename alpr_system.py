

import cv2
import numpy as np
import torch
import re
import os
import argparse
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time


def get_weights_path(model_name: str) -> str:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, "weights")
    model_path = os.path.join(weights_dir, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    return model_path


class StandaloneALPR:

    def __init__(self, 
                 plate_weight_path: str = None,
                 device: str = "auto",
                 plate_conf: float = 0.25,
                 ocr_threshold: float = 0.9):

        self.device = self._resolve_device(device)
        self.plate_conf = plate_conf
        self.ocr_threshold = ocr_threshold
        
        # Set default path to original project weights
        if plate_weight_path is None:
            plate_weight_path = get_weights_path("plate_yolov8n_320_2024.pt")
            
        self.plate_weight_path = plate_weight_path
        
        # Initialize models
        self._load_models()
        
        # Colors for visualization
        self.colors = {
            "plate": (0, 0, 255),    # Red
            "text": (255, 255, 255)  # White
        }
        
    def _resolve_device(self, requested: str) -> str:
        """Resolve device string"""
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requested.isdigit():
            return f"cuda:{requested}" if torch.cuda.is_available() else "cpu"
        else:
            return requested
            
    def _load_models(self):
        """Load YOLO models and OCR"""
        try:
            print(f"Loading plate detector from: {self.plate_weight_path}")
            self.plate_detector = YOLO(self.plate_weight_path)
            self.plate_detector.to(self.device)
            
            print("Loading OCR model...")
            # Initialize OCR without use_gpu parameter for compatibility
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en'
            )
            
            print(f"Models loaded successfully on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    def detect_plates(self, image: np.ndarray) -> List[Dict]:

        results = self.plate_detector(
            image,
            conf=self.plate_conf, 
            device=self.device,
            verbose=False
        )
        
        plates = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                
                plate = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                }
                plates.append(plate)
                
        return plates
        
    def extract_plate_text(self, image: np.ndarray, plate_bbox: List[int]) -> Tuple[str, float]:

        x1, y1, x2, y2 = plate_bbox
        
        # Crop plate region with some padding
        h, w = image.shape[:2]
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(w, x2 + 5)
        y2 = min(h, y2 + 5)
        
        plate_img = image[y1:y2, x1:x2]
        
        if plate_img.size == 0:
            return "", 0.0
            
        try:
            # Run OCR
            result = self.ocr.ocr(plate_img, cls=True)
            
            if result and result[0]:
                texts = []
                confidences = []
                
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                        
                        # Clean text - keep only alphanumeric and basic chars
                        text = re.sub(r'[^A-Za-z0-9\-.]', '', text)
                        
                        if text and len(text) > 1:
                            texts.append(text)
                            confidences.append(conf)
                
                if texts:
                    plate_text = ' '.join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    # Basic Vietnamese plate format correction
                    if len(plate_text) >= 3 and plate_text[0].isalpha() and plate_text[2] == 'C':
                        plate_text = plate_text[:2] + '0' + plate_text[3:]
                    
                    return plate_text, avg_confidence
                    
        except Exception as e:
            print(f"OCR error: {e}")
            
        return "", 0.0
        
    def process_image(self, image: np.ndarray, draw_results: bool = True) -> Dict:

        start_time = time.time()
        
        # Detect plates in the whole image
        plates = self.detect_plates(image)
        
        results = {
            'plates': [],
            'processing_time': 0,
            'image': image.copy() if draw_results else None
        }
        
        output_image = image.copy() if draw_results else None
        
        for plate in plates:
            plate_data = plate.copy()
            
            # Extract plate text
            plate_text, text_conf = self.extract_plate_text(image, plate['bbox'])
            plate_data['text'] = plate_text
            plate_data['text_confidence'] = text_conf
            
            # Only keep plates with reasonable confidence
            if text_conf >= self.ocr_threshold or len(plate_text) > 3:
                results['plates'].append(plate_data)
                
                # Draw plate detection
                if draw_results and output_image is not None:
                    px1, py1, px2, py2 = plate['bbox']
                    cv2.rectangle(output_image, (px1, py1), (px2, py2), self.colors['plate'], 2)
                    
                    # Draw plate text
                    if plate_text:
                        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(output_image, (px1, py1-25), (px1+text_size[0], py1), self.colors['plate'], -1)
                        cv2.putText(output_image, plate_text, (px1, py1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            
        results['processing_time'] = time.time() - start_time
        results['image'] = output_image
        
        return results
        
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):

        # Open video
        if isinstance(video_path, int) or video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)
            
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height} @ {fps}fps")
        
        # Setup video writer if saving
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                results = self.process_image(frame, draw_results=True)
                output_frame = results['image']
                
                # Add frame info
                info_text = f"Frame: {frame_count} | Time: {results['processing_time']:.3f}s"
                cv2.putText(output_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Print detection results
                if results['plates']:
                    for i, plate in enumerate(results['plates']):
                        if plate['text']:
                            print(f"Frame {frame_count} - Plate {i}: {plate['text']} (conf: {plate['text_confidence']:.3f})")
                
                # Save frame
                if writer:
                    writer.write(output_frame)
                    
                # Display frame
                if display:
                    cv2.imshow('ALPR System', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
                
            total_time = time.time() - start_time
            print(f"\nProcessed {frame_count} frames in {total_time:.2f}s")
            print(f"Average FPS: {frame_count/total_time:.2f}")


def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="Standalone ALPR System")
    parser.add_argument("--input", type=str, required=True,
                      help="Input image/video path or camera index (0 for webcam)")
    parser.add_argument("--output", type=str, 
                      help="Output path for video (optional)")
    parser.add_argument("--plate_model", type=str, 
                      default=None,  # Will use automatic path detection
                      help="Path to plate detection model")
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to run on (auto, cpu, cuda:0)")
    parser.add_argument("--plate_conf", type=float, default=0.25,
                      help="Plate detection confidence") 
    parser.add_argument("--ocr_threshold", type=float, default=0.9,
                      help="OCR confidence threshold")
    parser.add_argument("--no_display", action="store_true",
                      help="Don't display output (for headless mode)")
    
    args = parser.parse_args()
    
    # Initialize ALPR system
    alpr = StandaloneALPR(
        plate_weight_path=args.plate_model,
        device=args.device,
        plate_conf=args.plate_conf,
        ocr_threshold=args.ocr_threshold
    )
    
    # Check if input is image or video
    input_path = args.input
    
    # Handle webcam case
    if input_path.isdigit():
        print("Processing webcam...")
        alpr.process_video(int(input_path), args.output, not args.no_display)
        return
        
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
        
    # Determine if image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext in image_extensions:
        print("Processing image...")
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Cannot load image: {input_path}")
            return
            
        results = alpr.process_image(image, draw_results=True)
        
        # Print results
        print(f"\nDetection Results:")
        print(f"Processing time: {results['processing_time']:.3f}s")
        print(f"Plates found: {len(results['plates'])}")
        
        for i, plate in enumerate(results['plates']):
            if plate['text']:
                print(f"Plate {i+1}: '{plate['text']}' (conf: {plate['text_confidence']:.3f})")
        
        # Show/save result
        if not args.no_display:
            cv2.imshow('ALPR Result', results['image'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        if args.output:
            cv2.imwrite(args.output, results['image'])
            print(f"Result saved to: {args.output}")
            
    elif file_ext in video_extensions:
        print("Processing video...")
        alpr.process_video(input_path, args.output, not args.no_display)
        
    else:
        print(f"Error: Unsupported file format: {file_ext}")


if __name__ == "__main__":
    main()