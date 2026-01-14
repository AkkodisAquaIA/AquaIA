"""YOLO11 Inference Script for AquaIA Project

This script performs inference using YOLO11 model for macro invertebrate detection.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch


class YOLO11Predictor:
    """YOLO11 model predictor for object detection."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, device: str = 'cuda'):
        """
        Initialize YOLO11 predictor.
        
        Args:
            model_path: Path to the trained YOLO11 model (.pt file)
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {device}")
    
    def predict(self, image_path: str, save_results: bool = True, output_dir: str = 'predictions'):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            save_results: Whether to save annotated results
            output_dir: Directory to save results
            
        Returns:
            Detection results
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device,
            save=save_results,
            project=output_dir
        )
        
        return results
    
    def predict_batch(self, image_dir: str, save_results: bool = True, output_dir: str = 'predictions'):
        """
        Run inference on multiple images in a directory.
        
        Args:
            image_dir: Directory containing input images
            save_results: Whether to save annotated results
            output_dir: Directory to save results
            
        Returns:
            List of detection results
        """
        image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        
        results = []
        for img_path in image_paths:
            result = self.predict(str(img_path), save_results, output_dir)
            results.append(result)
        
        return results


def main():
    """Main function for running inference."""
    # Example usage
    model_path = 'path/to/your/yolo11_model.pt'
    image_path = 'path/to/your/image.jpg'
    
    # Initialize predictor
    predictor = YOLO11Predictor(
        model_path=model_path,
        conf_threshold=0.25,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run prediction
    results = predictor.predict(image_path)
    
    # Print results
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects")
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            print(f"Class: {class_id}, Confidence: {confidence:.2f}, BBox: {bbox}")


if __name__ == '__main__':
    main()
