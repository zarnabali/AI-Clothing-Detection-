"""
Main application entry point for the AI-Powered Clothing & Color Detection System
"""
import os
import argparse
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils.color_detection import extract_dominant_colors
from utils.visualization import draw_detections, convert_yolo_to_bbox, plot_results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clothing and Color Detection System")
    parser.add_argument("--image", type=str, help="Path to the input image file")
    parser.add_argument("--model", type=str, default="models/clothing_detector_final.pt", 
                       help="Path to the trained model file")
    parser.add_argument("--conf", type=float, default=0.3, 
                       help="Confidence threshold for detections")
    parser.add_argument("--save", action="store_true", 
                       help="Save the output image")
    parser.add_argument("--output_dir", type=str, default="output", 
                       help="Directory to save output images")
    
    args = parser.parse_args()
    
    # Check if image path is provided
    if not args.image:
        print("Error: Please provide an input image path using --image")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please train the model first or specify a valid model path.")
        return
    
    # Create output directory if it doesn't exist and save option is enabled
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(args.model)
    
    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image from {args.image}")
        return
    
    # Run inference
    results = model(image)
    
    # Convert YOLO results to a list of detections
    detections = convert_yolo_to_bbox(results, conf_threshold=args.conf)
    
    # Extract dominant colors for each clothing item
    color_info = {}
    for idx, detection in enumerate(detections):
        class_id, confidence, x1, y1, x2, y2 = detection
        colors = extract_dominant_colors(image, [x1, y1, x2, y2])
        color_info[idx] = colors
    
    # Draw detections and color information on the image
    result_image = draw_detections(image, detections, color_info)
    
    # Save the output image if requested
    if args.save:
        base_name = os.path.basename(args.image)
        output_path = os.path.join(args.output_dir, f"result_{base_name}")
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")
    
    # Display the results
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Clothing Detection and Color Analysis')
    plt.show()
    
    # Create a more detailed visualization
    fig = plot_results(image, detections, color_info)
    
    # Save the detailed visualization if requested
    if args.save:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output_dir, f"detailed_{base_name}.png")
        fig.savefig(output_path)
        print(f"Detailed result saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()