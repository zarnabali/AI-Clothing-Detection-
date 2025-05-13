"""
Real-time webcam demo for the AI-Powered Clothing & Color Detection System
"""
import cv2
import argparse
import time
import numpy as np
import torch
import os
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.nn.modules.block import DFL, Proto, Bottleneck
from ultralytics.nn.modules.conv import Conv2, LightConv, DWConv, DWConvTranspose2d, ConvTranspose, Focus, GhostConv, RepConv, Concat
from torch.nn.modules.container import Sequential, ModuleList
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU, ReLU, LeakyReLU
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.linear import Linear
from utils.color_detection import extract_dominant_colors, get_color_name

# Add required classes to safe globals
torch.serialization.add_safe_globals([
    DetectionModel,
    Conv, C2f, SPPF, Detect,
    DFL, Proto, Bottleneck,
    Conv2, LightConv, DWConv, DWConvTranspose2d, ConvTranspose, Focus, GhostConv, RepConv, Concat,
    Sequential, ModuleList,
    Conv2d,
    BatchNorm2d,
    SiLU, ReLU, LeakyReLU,
    MaxPool2d, AdaptiveAvgPool2d,
    Upsample,
    Dropout,
    LayerNorm,
    Linear
])

# Force unsafe loading for compatibility with all PyTorch versions
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

# For torch.load directly
import functools
try:
    # Monkey patch torch.load to always use weights_only=False
    original_torch_load = torch.load
    torch.load = functools.partial(original_torch_load, weights_only=False)
    print("✅ Modified torch.load to use weights_only=False by default")
except Exception as e:
    print(f"Note: Could not patch torch.load: {e}")

# Define clothing classes
CLOTHING_CLASSES = {
    0: "Dress",
    1: "Hoodie",
    2: "Dress Shirt",
    3: "Shorts",
    4: "Skirt",
    5: "Shirt",
    6: "Pants",
    7: "Jacket"
}

def draw_detections(image, detections, color_info):
    """
    Draw bounding boxes and color information on the image.
    
    Args:
        image: Input image
        detections: List of detections [class_id, confidence, x1, y1, x2, y2]
        color_info: Dictionary mapping detection indices to color information
    """
    result_image = image.copy()
    
    for idx, detection in enumerate(detections):
        class_id, confidence, x1, y1, x2, y2 = detection
        
        # Get the dominant color for this detection
        if idx in color_info and color_info[idx]:
            # Convert RGB to BGR for OpenCV
            dominant_color = color_info[idx][0]  # Get the most dominant color
            # Ensure color is a tuple of integers
            bgr_color = (int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0]))  # RGB to BGR
            color_name = get_color_name(dominant_color)
        else:
            bgr_color = (255, 255, 255)  # Default to white in BGR
            color_name = "Unknown"
        
        # Get class name
        class_name = CLOTHING_CLASSES.get(class_id, f"Class {class_id}")
        
        # Draw bounding box
        cv2.rectangle(result_image, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     bgr_color, 2)
        
        # Create label with class name and color
        label = f"{class_name} - {color_name} ({confidence:.2f})"
        
        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(label, 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                                       0.5, 2)
        cv2.rectangle(result_image, 
                     (int(x1), int(y1) - label_height - 10),
                     (int(x1) + label_width, int(y1)),
                     bgr_color, 
                     -1)
        
        # Draw label text
        cv2.putText(result_image, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_image

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clothing and Color Detection Webcam Demo")
    parser.add_argument("--model", type=str, default="models/clothing_detector_20250511_131005.pt", 
                       help="Path to the trained model file")
    parser.add_argument("--conf", type=float, default=0.3, 
                       help="Confidence threshold for detections")
    parser.add_argument("--device", type=int, default=0, 
                       help="Webcam device ID")
    parser.add_argument("--width", type=int, default=640, 
                       help="Width of the webcam capture")
    parser.add_argument("--height", type=int, default=480, 
                       help="Height of the webcam capture")
    parser.add_argument("--fps_limit", type=int, default=15, 
                       help="FPS limit for the webcam feed")
    args = parser.parse_args()
    
    # Load the model
    try:
        model = YOLO(args.model)
        print(f"Model loaded successfully from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open webcam
    try:
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam with device ID {args.device}")
            return
            
        print(f"Webcam opened successfully with resolution {args.width}x{args.height}")
    except Exception as e:
        print(f"Error opening webcam: {e}")
        return
    
    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    # Variables for detection optimization
    detection_interval = 1.0 / args.fps_limit  # seconds between detections
    last_detection_time = 0
    last_detections = []
    last_color_info = {}
    
    print("\nStarting webcam demo...")
    print("Press 'q' to quit")
    print("Press 's' to save the current frame")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam")
                break
            
            # Calculate current FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            # Check if it's time to run detection again
            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                # Run model inference
                results = model(frame)
                
                # Process results
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if box.conf[0] >= args.conf:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            class_id = int(box.cls[0].item())
                            confidence = box.conf[0].item()
                            detections.append([class_id, confidence, x1, y1, x2, y2])
                
                # Extract dominant colors for each clothing item
                color_info = {}
                for idx, detection in enumerate(detections):
                    class_id, confidence, x1, y1, x2, y2 = detection
                    try:
                        colors = extract_dominant_colors(frame, [x1, y1, x2, y2])
                        color_info[idx] = colors
                    except Exception as e:
                        print(f"Error extracting colors: {e}")
                
                # Update stored detections and colors
                last_detections = detections
                last_color_info = color_info
                last_detection_time = current_time
            
            # Draw detections and color information on the frame
            result_frame = draw_detections(frame, last_detections, last_color_info)
            
            # Add FPS counter
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame to a temporary file and display it
            temp_file = "temp_frame.jpg"
            cv2.imwrite(temp_file, result_frame)
            
            # Display the frame using the default image viewer
            os.system(f'start {temp_file}')
            
            # Handle key presses
            key = input("Press 'q' to quit, 's' to save frame, or Enter to continue: ").lower()
            if key == 'q':
                break
            elif key == 's':
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join("output", f"webcam_capture_{timestamp}.jpg")
                cv2.imwrite(filename, result_frame)
                print(f"Saved frame to {filename}")
            
            # Remove temporary file
            try:
                os.remove(temp_file)
            except:
                pass
    
    finally:
        # Release resources
        cap.release()
        print("Webcam demo stopped.")

if __name__ == "__main__":
    main()