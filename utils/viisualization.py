"""
Visualization utilities for displaying detection results
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .color_detection import name_color

# Define colors for different clothing classes (BGR format)
CLASS_COLORS = {
    0: (0, 255, 0),     # Upper clothes - Green
    1: (255, 0, 0),     # Lower clothes - Blue
    2: (0, 0, 255)      # Shoes - Red
}

# Class names mapping
CLASS_NAMES = {
    0: "Upper Clothes",
    1: "Lower Clothes",
    2: "Shoes"
}

def draw_detections(image, detections, color_info=None, display_colors=True):
    """
    Draw bounding boxes and color information on the image
    
    Args:
        image: Input image
        detections: List of detection results with format [class_id, confidence, x1, y1, x2, y2]
        color_info: Dictionary mapping bbox indices to color information
        display_colors: Whether to display color information
    
    Returns:
        Image with annotations
    """
    # Create a copy of the image
    result_image = image.copy()
    
    # Draw each detection
    for idx, detection in enumerate(detections):
        class_id, confidence, x1, y1, x2, y2 = detection
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get the color for this class
        color = CLASS_COLORS.get(int(class_id), (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text = f"{CLASS_NAMES.get(int(class_id), 'Unknown')} {confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(result_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add color information if available
        if display_colors and color_info and idx in color_info:
            y_offset = y1 + 20
            
            for color_idx, color_data in enumerate(color_info[idx][:3]):  # Display top 3 colors
                rgb_color = color_data['rgb']
                hsv_color = color_data['hsv']
                percentage = color_data['percentage']
                color_name = name_color(hsv_color)
                
                # Create color swatch
                cv2.rectangle(
                    result_image, 
                    (x2 + 10, y_offset), 
                    (x2 + 40, y_offset + 20), 
                    (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0])), 
                    -1
                )
                
                # Add color name and percentage
                color_text = f"{color_name}: {percentage:.1f}%"
                cv2.putText(
                    result_image, 
                    color_text, 
                    (x2 + 45, y_offset + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (0, 0, 0), 
                    1
                )
                
                y_offset += 25
    
    return result_image

def convert_yolo_to_bbox(results, conf_threshold=0.3):
    """
    Convert YOLOv8 results to a list of detections
    
    Args:
        results: YOLOv8 results object
        conf_threshold: Confidence threshold for filtering detections
    
    Returns:
        List of detections with format [class_id, confidence, x1, y1, x2, y2]
    """
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box data
            conf = box.conf.cpu().numpy()[0]
            
            # Filter by confidence
            if conf < conf_threshold:
                continue
                
            cls = int(box.cls.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]  # x1, y1, x2, y2
            
            detections.append([cls, conf, *xyxy])
    
    return detections

def plot_results(image, detections, color_info, figsize=(12, 10)):
    """
    Plot detection results with color information
    
    Args:
        image: Input image
        detections: List of detection results
        color_info: Color information for each detection
        figsize: Figure size
    """
    # Create figure
    fig, axes = plt.subplots(1 + len(detections), 1, figsize=figsize)
    
    # If only one detection, make axes a list
    if len(detections) == 1:
        axes = [axes]
    else:
        # Show the main image on the first axis
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image with Detections')
        axes[0].axis('off')
        
        # Use the remaining axes for the crops
        axes = axes[1:]
    
    # Display each detection crop and its colors
    for i, detection in enumerate(detections):
        class_id, confidence, x1, y1, x2, y2 = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Extract the region
        crop = image[y1:y2, x1:x2]
        
        # Convert to RGB for display
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Get the class name
        class_name = CLASS_NAMES.get(int(class_id), 'Unknown')
        
        # Display the crop
        axes[i].imshow(crop_rgb)
        
        # Display color information
        if i in color_info:
            colors_text = ", ".join([
                f"{name_color(c['hsv'])} ({c['percentage']:.1f}%)" 
                for c in color_info[i][:3]
            ])
            axes[i].set_title(f"{class_name} (conf: {confidence:.2f})\nColors: {colors_text}")
        else:
            axes[i].set_title(f"{class_name} (conf: {confidence:.2f})")
            
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig