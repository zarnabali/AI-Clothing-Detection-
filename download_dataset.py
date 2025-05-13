"""
Download fashion detection dataset from Roboflow
"""
# Install the Roboflow package if you haven't already
# You can run this in terminal before running this script:
# pip install roboflow

from roboflow import Roboflow

def download_fashion_dataset():
    """Download the fashion detection dataset from Roboflow"""
    # Import and authenticate with Roboflow
    print("Authenticating with Roboflow...")
    rf = Roboflow(api_key="BpiN0KWHY6TZA81iDlTX")
    
    # Access the project and download the dataset
    print("Accessing fashion detection project...")
    project = rf.workspace("penelitian-xr2mg").project("fashion-detection-bounding-box")
    version = project.version(1)
    
    print("Downloading dataset (this might take a while)...")
    dataset = version.download("yolov8")
    
    print(f"Dataset downloaded successfully to: {dataset.location}")
    print("\nNext steps:")
    print("1. Update the 'path' in clothing_dataset.yaml to point to this location")
    print("2. Run model_training.py to train your model")

if __name__ == "__main__":
    download_fashion_dataset()