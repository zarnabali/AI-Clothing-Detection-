import cv2
import numpy as np

def extract_dominant_colors(image_region, n=1):
    """
    Extract dominant colors from an image region.
    
    Args:
        image_region (numpy.ndarray): The image region to analyze
        n (int): Number of dominant colors to extract
        
    Returns:
        list: List of dominant colors as BGR tuples
    """
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
    """
    Get the name of a color from its BGR value.
    
    Args:
        bgr_color (tuple): BGR color tuple
        
    Returns:
        str: Name of the color
    """
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
    elif max(r, g, b) - min(r, g, b) < 30:
        return "Gray"
    elif r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    else:
        return "Unknown"