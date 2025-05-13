# AI Clothing Detection Model

A real-time clothing detection system using YOLOv8 that can detect and classify different types of clothing items in images and video streams.

## Features

- Real-time clothing detection using webcam
- Image upload and processing
- Detection of multiple clothing items including:
  - Dresses
  - Hoodies
  - Dress Shirts
  - Shorts
  - Skirts
  - Shirts
  - Pants
  - Jackets
- Color detection for each clothing item
- Download processed images
- Modern and responsive web interface

## Requirements

- Python 3.8+
- Flask
- OpenCV
- PyTorch
- Ultralytics YOLOv8

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zarnabali/AI-Clothing-Detection-Model.git
cd AI-Clothing-Detection-Model
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
```bash
# The model will be downloaded automatically when running the application
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Upload images for detection
   - Use real-time webcam detection
   - View detection results
   - Download processed images

## Project Structure

```
AI-Clothing-Detection-Model/
├── app.py              # Main Flask application
├── models/            # Model directory
├── templates/         # HTML templates
├── static/           # Static files (CSS, JS)
├── uploads/          # Temporary upload directory
└── output/           # Processed images directory
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- Flask web framework
- OpenCV for image processing 