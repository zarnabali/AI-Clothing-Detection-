from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import cv2
import os
import time
import numpy as np
import torch
import sys
import traceback

print("Starting application...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Current working directory: {os.getcwd()}")

# Import necessary modules
print("Importing modules...")
from torch.serialization import add_safe_globals, safe_globals
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import *
from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv, Concat

# Import more modules from ultralytics - including DFL
try:
    print("Importing DFL module...")
    from ultralytics.nn.modules.block import DFL
    print("DFL module imported successfully")
except ImportError:
    print("DFL module not found in ultralytics.nn.modules.block")
    # DFL might be in a different location in the newer ultralytics version
    try:
        print("Searching for DFL in other locations...")
        # Try to import from all possible locations
        from ultralytics.nn.modules import DFL
        print("DFL found in ultralytics.nn.modules")
    except ImportError:
        print("DFL module not found in ultralytics.nn.modules")
        print("Creating a placeholder DFL class")
        # Create a placeholder DFL class to register with safe_globals
        class DFL(torch.nn.Module):
            """Placeholder for DFL class"""
            pass

print(f"Ultralytics version: {ultralytics.__version__}")

# Register all potentially needed classes as safe
print("Registering safe classes...")
classes_to_add = [
    # Your original classes
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.Concat,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.SPPF,
    ultralytics.nn.modules.block.Bottleneck,
    ultralytics.nn.modules.head.Detect,
    DetectionModel,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.dropout.Dropout,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.pooling.MaxPool2d,
    ultralytics.nn.tasks.DetectionModel,
    torch.nn.modules.upsampling.Upsample,
    
    # Add the DFL class that was reported missing
    DFL
]

# Add more potentially needed classes
try:
    print("Adding more torch and ultralytics classes to safe list...")
    # Add more torch module classes that might be needed
    for module_name in dir(torch.nn.modules):
        if not module_name.startswith('_'):
            try:
                module = getattr(torch.nn.modules, module_name)
                if isinstance(module, type) and issubclass(module, torch.nn.Module):
                    classes_to_add.append(module)
            except (TypeError, AttributeError):
                pass
    
    # Add common ultralytics classes
    for module_name in ['Conv', 'Concat', 'C2f', 'C3', 'SPPF', 'Bottleneck', 'Detect', 'DetectionModel']:
        if hasattr(ultralytics.nn.modules, module_name):
            classes_to_add.append(getattr(ultralytics.nn.modules, module_name))
except Exception as e:
    print(f"Error adding extra classes: {str(e)}")

# Add classes to safe globals
print("Adding classes to safe globals...")
try:
    add_safe_globals(classes_to_add)
    print("Safe globals added successfully")
except Exception as e:
    print(f"Error adding safe globals: {str(e)}")
    traceback.print_exc()

# Monkey patch torch.load to use weights_only=False by default
# This is a temporary solution that should be used with caution
print("Applying monkey patch to torch.load...")
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    """Patched version of torch.load that uses weights_only=False by default"""
    if 'weights_only' not in pickle_load_args:
        print("Using weights_only=False for torch.load")
        pickle_load_args['weights_only'] = False
    return original_torch_load(f, map_location, pickle_module, **pickle_load_args)

# Apply the monkey patch
torch.load = patched_torch_load
print("torch.load has been patched to use weights_only=False by default")

# FIX: Import color detection utilities safely
print("Importing color detection modules...")
try:
    from utils.color_detection import extract_dominant_colors, get_color_name
    print("Color detection modules imported successfully")
except ImportError as e:
    print(f"Error importing color detection modules: {str(e)}")
    # Create fallback color detection functions
    def extract_dominant_colors(image_region, n=1):
        """Fallback function to extract dominant colors from an image region."""
        if image_region is None or image_region.size == 0:
            return [(0, 0, 255)]  # Default blue color

        # Reshape the image for K-means
        pixels = image_region.reshape(-1, 3)
        pixels = pixels.astype(np.float32)
        
        # Use K-means to find dominant colors
        if len(pixels) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            try:
                _, labels, palette = cv2.kmeans(pixels, n, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)
                
                # Convert to BGR format and return as list of tuples
                return [tuple(map(int, color)) for color in palette]
            except Exception as e:
                print(f"Error in K-means: {str(e)}")
                return [(0, 0, 255)]  # Default blue color
        return [(0, 0, 255)]  # Default blue color
    
    def get_color_name(bgr_color):
        """Fallback function to get color name from BGR value."""
        # Convert BGR to RGB
        r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]
        
        # Define basic color ranges
        if r > 200 and g < 100 and b < 100:
            return "Red"
        elif r > 200 and g > 200 and b < 100:
            return "Yellow"
        elif r < 100 and g > 200 and b < 100:
            return "Green"
        elif r < 100 and g < 100 and b > 200:
            return "Blue"
        elif r > 200 and g > 100 and b > 200:
            return "Purple"
        elif r > 200 and g > 150 and b > 150:
            return "Pink"
        elif r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 100 and g < 100 and b < 100:
            return "Black"
        else:
            return "Gray"
    
    print("Created fallback color detection functions")

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure static folder for processed images
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLO model
model_path = "models/clothing_detector_20250511_131005.pt"
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")

def load_model():
    """Load the YOLO model with proper error handling"""
    print("Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading custom model: {str(e)}")
        traceback.print_exc()
        
        # Plan B: Use a base model if custom model fails
        print("Falling back to base YOLOv8n model")
        try:
            model = YOLO('yolov8n.pt')
            print("Base model loaded successfully")
            return model
        except Exception as e2:
            print(f"Error loading base model: {str(e2)}")
            traceback.print_exc()
            print("All model loading attempts failed")
            sys.exit(1)

# Load the model
try:
    model = load_model()
except Exception as e:
    print(f"Unhandled error in model loading: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Define clothing classes
CLOTHING_CLASSES = {
    0: 'Dress',
    1: 'Hoodie',
    2: 'Dress Shirt',
    3: 'Shorts',
    4: 'Skirt',
    5: 'Shirt',
    6: 'Pants',
    7: 'Jacket'
}

def process_image(image):
    """Process an image and return detection results."""
    print("Processing image...")
    try:
        # Run YOLO detection
        results = model.predict(image, conf=0.25)[0]  # Get first result
        print(f"Detection complete, found {len(results.boxes)} detections")
        
        detections = []
        if hasattr(results, 'boxes'):
            boxes = results.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                print(f"Detected {CLOTHING_CLASSES.get(class_id, 'Unknown')} with confidence {confidence:.2f}")
                
                # Extract the clothing region
                clothing_region = image[y1:y2, x1:x2]
                if clothing_region.size > 0:  # Check if region is valid
                    # Get dominant colors
                    colors = extract_dominant_colors(clothing_region)
                    color_name = get_color_name(colors[0]) if colors else "Unknown"
                    print(f"Dominant color: {color_name}")
                    
                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{CLOTHING_CLASSES.get(class_id, 'Unknown')} ({color_name})"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detections.append({
                        'class': CLOTHING_CLASSES.get(class_id, 'Unknown'),
                        'color': color_name,
                        'confidence': f"{confidence:.2f}",
                        'box': [x1, y1, x2, y2]
                    })
        
        print(f"Processed image with {len(detections)} detections")
        return image, detections
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return image, []

def generate_frames():
    """Generate video frames with real-time detection."""
    print("Starting webcam capture...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print status every 30 frames
            print(f"Captured frame {frame_count}")
        
        try:
            # Process frame
            processed_frame, detections = process_image(frame)
            if detections:
                print(f"Detected {len(detections)} objects in frame {frame_count}")
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            traceback.print_exc()
            continue
    
    print("Releasing webcam")
    cap.release()

@app.route('/')
def index():
    """Render the main page."""
    print("Serving index page")
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    """Render the webcam page."""
    print("Serving webcam page")
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    print("Starting video feed")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and processing."""
    print("File upload requested")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Save the uploaded file
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            print(f"File saved to {filename}")
            
            # Read and process the image
            image = cv2.imread(filename)
            if image is None:
                print(f"Failed to read image from {filename}")
                return jsonify({'error': 'Failed to read image'})
            
            print(f"Image read successfully, shape: {image.shape}")
            
            # Process the image
            processed_image, detections = process_image(image)
            print(f"Found {len(detections)} detections")
            
            # Save the processed image
            output_filename = os.path.join(OUTPUT_FOLDER, f"processed_{file.filename}")
            cv2.imwrite(output_filename, processed_image)
            print(f"Processed image saved to {output_filename}")
            
            # Return results with the correct image path
            return jsonify({
                'image_path': f"/output/processed_{file.filename}",
                'detections': detections,
                'message': f"Found {len(detections)} clothing items"
            })
        except Exception as e:
            print(f"Error processing upload: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/output/<filename>')
def serve_processed_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)