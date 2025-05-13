"""
Script to test the trained clothing detection model on a single image
"""
import os
import sys

# Force NumPy to use 1.x API
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# Downgrade numpy to 1.x version for compatibility
print("Installing compatible NumPy version...")
os.system("pip install numpy==1.24.3")

# Downgrade torch to a compatible version
print("Installing compatible PyTorch version...")
os.system("pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118")

# Make sure ultralytics is installed with exact version
print("Installing ultralytics...")
os.system("pip install ultralytics==8.0.196")

# Now import the required libraries
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from PIL import Image

# Add required classes to safe globals
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
print("✅ Set TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 for PyTorch compatibility")

# For torch.load directly
import functools
try:
    # Monkey patch torch.load to always use weights_only=False
    original_torch_load = torch.load
    torch.load = functools.partial(original_torch_load, weights_only=False)
    print("✅ Modified torch.load to use weights_only=False by default")
except Exception as e:
    print(f"Note: Could not patch torch.load: {e}")

def detect_dominant_color(image, bbox):
    """Detect the dominant color in the bounding box region with improved accuracy"""
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    
    # Convert to RGB if image is BGR
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better color analysis
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    
    # Reshape the ROI to a list of pixels
    pixels = hsv_roi.reshape(-1, 3)
    
    # Remove any pure white, black, or very dark pixels (likely background or shadows)
    mask = ~(
        (pixels[:, 1] < 20) |  # Low saturation
        (pixels[:, 2] < 20) |  # Low value (dark)
        (pixels[:, 2] > 240)   # Very bright
    )
    pixels = pixels[mask]
    
    if len(pixels) == 0:
        return np.array([128, 128, 128])  # Return gray if no valid pixels
    
    # Convert to float32 for k-means
    pixels = np.float32(pixels)
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    k = 5  # Number of clusters for better color separation
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Get the dominant color (the one with most pixels)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = centers[np.argmax(counts)]
    
    # Convert back to RGB for display
    rgb_color = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_HSV2RGB)[0][0]
    
    return rgb_color

def get_color_name(rgb_color):
    """Convert RGB color to a human-readable color name with improved accuracy"""
    # Convert RGB to HSV for better color comparison
    rgb_color = np.array(rgb_color)
    hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # Define HSV ranges for each color with more precise boundaries
    color_ranges = {
        'Black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 30, 30])},
        'White': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 20, 255])},
        'Red': [
            {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            {'lower': np.array([170, 100, 100]), 'upper': np.array([180, 255, 255])}
        ],
        'Green': {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])},
        'Blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([140, 255, 255])},
        'Yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([40, 255, 255])},
        'Cyan': {'lower': np.array([80, 100, 100]), 'upper': np.array([100, 255, 255])},
        'Magenta': {'lower': np.array([140, 100, 100]), 'upper': np.array([160, 255, 255])},
        'Gray': {'lower': np.array([0, 0, 50]), 'upper': np.array([180, 30, 200])},
        'Brown': {'lower': np.array([10, 100, 20]), 'upper': np.array([20, 255, 200])},
        'Orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([20, 255, 255])},
        'Purple': {'lower': np.array([130, 100, 100]), 'upper': np.array([140, 255, 255])},
        'Pink': {'lower': np.array([150, 50, 100]), 'upper': np.array([170, 255, 255])},
        'Beige': {'lower': np.array([20, 30, 200]), 'upper': np.array([40, 100, 255])},
        'Navy': {'lower': np.array([100, 100, 50]), 'upper': np.array([140, 255, 100])},
        'Maroon': {'lower': np.array([0, 100, 50]), 'upper': np.array([10, 255, 100])},
        'Olive': {'lower': np.array([40, 100, 50]), 'upper': np.array([60, 255, 100])},
        'Teal': {'lower': np.array([80, 100, 50]), 'upper': np.array([100, 255, 100])},
        'Burgundy': {'lower': np.array([0, 100, 50]), 'upper': np.array([10, 255, 80])},
        'Khaki': {'lower': np.array([30, 30, 150]), 'upper': np.array([50, 100, 255])},
        'Lavender': {'lower': np.array([130, 50, 150]), 'upper': np.array([150, 100, 255])},
        'Mint': {'lower': np.array([60, 50, 150]), 'upper': np.array([80, 100, 255])},
        'Coral': {'lower': np.array([5, 100, 100]), 'upper': np.array([15, 255, 255])},
        'Gold': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
        'Silver': {'lower': np.array([0, 0, 150]), 'upper': np.array([180, 30, 255])}
    }
    
    # Special handling for red (which wraps around in HSV)
    if 'Red' in color_ranges:
        red_ranges = color_ranges.pop('Red')
        for red_range in red_ranges:
            if (red_range['lower'] <= hsv_color).all() and (hsv_color <= red_range['upper']).all():
                return 'Red'
    
    # Check which color range the HSV value falls into
    for color_name, ranges in color_ranges.items():
        if isinstance(ranges, dict):  # Single range
            if (ranges['lower'] <= hsv_color).all() and (hsv_color <= ranges['upper']).all():
                return color_name
    
    # If no exact match, find the closest color using HSV distance
    min_dist = float('inf')
    closest_color = 'Unknown'
    
    # Calculate distance to each color's center
    for color_name, ranges in color_ranges.items():
        if isinstance(ranges, dict):  # Single range
            center = (ranges['lower'] + ranges['upper']) / 2
            # Weight the distance calculation to favor hue over saturation and value
            dist = np.sum([
                (hsv_color[0] - center[0]) ** 2 * 2,  # Hue is more important
                (hsv_color[1] - center[1]) ** 2,      # Saturation
                (hsv_color[2] - center[2]) ** 2       # Value
            ])
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
    
    return closest_color

def predict_clothing_with_colors(model_path, image_path):
    """Predict clothing items and their colors in the image"""
    try:
        # Load the model with weights_only=False
        model = YOLO(model_path)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Make prediction
        results = model(image)
        
        # Print header for detection results
        print("\n===== CLOTHING DETECTION RESULTS =====")
        
        # Process results
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                print("No clothing items detected in the image.")
                continue
                
            print(f"\nDetected {len(boxes)} clothing items:")
            
            for i, box in enumerate(boxes, 1):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Get class name
                class_name = result.names[class_id]
                
                # Detect dominant color
                color = detect_dominant_color(image, (x1, y1, x2, y2))
                
                # Convert RGB to color name
                color_name = get_color_name(color)
                
                # Print detection details
                print(f"\nItem {i}:")
                print(f"Class: {class_name}")
                print(f"Color: {color_name} (RGB: {color[0]},{color[1]},{color[2]})")
                print(f"Confidence: {conf:.2%}")
                
                # Draw bounding box with color-matched border
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color.tolist(), 2)
                
                # Create label with class name and color
                label = f"{class_name} - {color_name}"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, 
                            (int(x1), int(y1) - label_height - 10),
                            (int(x1) + label_width, int(y1)),
                            color.tolist(), 
                            -1)
                
                # Draw label text
                cv2.putText(image, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save the result
        output_path = "test_result1.jpg"
        cv2.imwrite(output_path, image)
        print(f"\nResult saved to {output_path}")
        
        # Display the image
        cv2.imshow("Clothing Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("\nTrying alternative loading method...")
        try:
            # Try loading with explicit weights_only=False
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            model = YOLO(model)
            print("Model loaded successfully with alternative method")
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")

def main():
    # Model and image paths
    model_path = "models\\clothing_detector_20250511_131005.pt"
    image_path = "datasets\\Fashion-Detection-(Bounding-Box)-1\\test\\images\\Pimage6342_jpg.rf.058528a289d092502676042c73538c57.jpg"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run prediction
    predict_clothing_with_colors(model_path, image_path)

if __name__ == "__main__":
    main() 