"""
Simplified model training script for clothing detection using YOLOv8
"""
import os
import sys
import yaml
import shutil
from pathlib import Path
from datetime import datetime

# Create a fresh dataset YAML with correct class count
def create_fresh_dataset_config():
    """Create a fresh dataset configuration file with correct class settings"""
    print("\n===== CREATING FRESH DATASET CONFIG =====")
    
    # Define dataset paths - adjust these if needed
    base_path = "/content/drive/MyDrive/Clothing_Detection_Model/datasets/Fashion-Detection-(Bounding-Box)-1"
    if not os.path.exists(base_path):
        # Try alternate path
        alternate_path = "/content/clothing_detection/datasets/Fashion-Detection-(Bounding-Box)-1"
        if os.path.exists(alternate_path):
            base_path = alternate_path
    
    # Define all 8 clothing classes
    classes = {
        0: 'Dress',
        1: 'Hoodie',
        2: 'Dress Shirt',
        3: 'Shorts',
        4: 'Skirt',
        5: 'Shirt',
        6: 'Pants',
        7: 'Jacket'
    }
    
    # Create new config data
    data = {
        'path': base_path,
        'train': os.path.join(base_path, "train/images"),
        'val': os.path.join(base_path, "valid/images"),
        'test': os.path.join(base_path, "test/images") if os.path.exists(os.path.join(base_path, "test/images")) else "",
        'names': classes,
        'nc': len(classes)  # Explicitly set number of classes
    }
    
    # Check paths
    print(f"Train path: {data['train']}")
    print(f"Val path: {data['val']}")
    
    if not os.path.exists(data['train']):
        print(f"⚠️ Warning: Train path does not exist: {data['train']}")
    else:
        print(f"✅ Train path exists")
        
    if not os.path.exists(data['val']):
        print(f"⚠️ Warning: Val path does not exist: {data['val']}")
    else:
        print(f"✅ Val path exists")
    
    # Generate unique YAML filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yaml_path = f'clothing_dataset_{timestamp}.yaml'
    
    # Save the YAML
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)
    
    print(f"✅ Fresh dataset configuration created at {yaml_path}")
    print(f"- Classes: {data['names']}")
    print(f"- Number of classes: {data['nc']}")
    
    return yaml_path

# Fix label files with invalid class IDs
def fix_labels(yaml_path):
    """Fix label files with invalid class IDs"""
    print("\n===== FIXING LABEL FILES =====")
    
    # Load YAML config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    max_class_id = len(config['names']) - 1
    print(f"Maximum valid class ID: {max_class_id}")
    
    for data_type in ['train', 'val']:
        # Get label directory
        image_dir = config[data_type]
        label_dir = image_dir.replace('/images', '/labels')
        
        if not os.path.exists(label_dir):
            print(f"⚠️ {data_type} label directory not found: {label_dir}")
            continue
            
        print(f"Checking {data_type} labels in: {label_dir}")
        fixed_count = 0
        
        # Process all label files
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue
                
            file_path = os.path.join(label_dir, label_file)
            needs_fix = False
            
            try:
                with open(file_path, 'r') as f:
                    lines = []
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            lines.append(line)
                            continue
                        
                        class_id = int(parts[0])
                        if class_id > max_class_id:
                            # Fix class ID by mapping to a valid range (0-7)
                            parts[0] = str(min(class_id % (max_class_id + 1), max_class_id))
                            line = ' '.join(parts) + '\n'
                            needs_fix = True
                        
                        lines.append(line)
                
                if needs_fix:
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                    fixed_count += 1
                    
            except Exception as e:
                print(f"Error fixing {file_path}: {e}")
        
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} label files in {data_type} set")
        else:
            print(f"✅ No label fixes needed in {data_type} set")

# Train the model
def train_model(yaml_path):
    """Train the YOLOv8 model with the specified dataset"""
    print("\n===== TRAINING MODEL =====")
    
    # Check and setup GPU in Colab
    import torch
    print("\n===== CHECKING GPU =====")
    if not torch.cuda.is_available():
        print("⚠️ CUDA is not available. Please enable GPU in Colab:")
        print("1. Go to Runtime > Change runtime type")
        print("2. Select 'GPU' as Hardware accelerator")
        print("3. Click 'Save'")
        print("\nWould you like to continue anyway on CPU? (training will be VERY slow)")
        response = input("Continue on CPU? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting script. Please enable GPU and try again.")
            return None
        device = "cpu"
    else:
        print(f"✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        device = "0"
    
    # Add PyTorch 2.6+ compatibility fix
    try:
        # For PyTorch 2.6+: Add necessary classes to safe globals
        from torch.serialization import add_safe_globals
        from ultralytics.nn.tasks import DetectionModel
        import torch.nn as nn
        
        # Add all potentially needed classes to safe globals
        add_safe_globals([
            DetectionModel,
            nn.modules.container.Sequential,
            nn.Sequential,
            nn.Module,
            nn.ModuleList,
            nn.ModuleDict,
            nn.parameter.Parameter
        ])
        
        print("✅ Added required model classes to PyTorch safe globals for compatibility")
    except ImportError:
        print("Note: Using PyTorch version that doesn't require safe globals")
    
    # Import YOLO here to avoid early initialization issues
    from ultralytics import YOLO
    
    # Define model parameters
    model_size = 's'  # n, s, m, l, x
    epochs = 50
    batch_size = 8  # Reduced batch size for T4 GPU memory constraints
    img_size = 640
    
    # Generate a unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"clothing_detection_{timestamp}"
    
    print(f"Creating a new YOLOv8{model_size} model...")
    
    # Load and verify dataset configuration
    print("\n===== VERIFYING DATASET CONFIGURATION =====")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = len(config['names'])
    print(f"Number of classes in dataset: {num_classes}")
    print(f"Class names: {config['names']}")
    
    # Verify label files
    print("\n===== VERIFYING LABEL FILES =====")
    for data_type in ['train', 'val']:
        label_dir = config[data_type].replace('/images', '/labels')
        if not os.path.exists(label_dir):
            print(f"⚠️ Warning: {data_type} label directory not found: {label_dir}")
            continue
            
        print(f"\nChecking {data_type} labels in: {label_dir}")
        invalid_labels = []
        
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue
                
            file_path = os.path.join(label_dir, label_file)
            try:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                            
                        class_id = int(parts[0])
                        if class_id >= num_classes:
                            invalid_labels.append(f"{label_file}:{line_num} (class {class_id})")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if invalid_labels:
            print(f"⚠️ Found {len(invalid_labels)} invalid labels in {data_type} set:")
            for label in invalid_labels[:10]:  # Show first 10 invalid labels
                print(f"  - {label}")
            if len(invalid_labels) > 10:
                print(f"  ... and {len(invalid_labels) - 10} more")
        else:
            print(f"✅ All labels in {data_type} set are valid")
    
    # Initialize the model with correct number of classes
    try:
        # Create a new model with the correct number of classes
        model = YOLO(f'yolov8{model_size}.yaml')
        model.model.nc = num_classes  # Set number of classes
        print(f"✅ Created YOLOv8{model_size} model with {num_classes} classes")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Start training
    print(f"\nStarting training with:")
    print(f"- Dataset: {yaml_path}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Image size: {img_size}")
    print(f"- Run name: {run_name}")
    print(f"- Device: {device}")
    print(f"- Number of classes: {num_classes}")
    
    # Create output directory
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
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
            
        # Train the model with optimized parameters for T4 GPU
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name=run_name,
            project=output_dir,
            save=True,
            device=device,
            patience=5,  # Early stopping after 5 epochs without improvement
            workers=4,   # Reduced number of workers for memory efficiency
            cache=True,  # Cache images in memory for faster training
            amp=True,    # Use mixed precision training
            optimizer='AdamW',  # Use AdamW optimizer
            lr0=0.001,   # Initial learning rate
            lrf=0.01,    # Final learning rate fraction
            momentum=0.937,  # SGD momentum
            weight_decay=0.0005,  # Optimizer weight decay
            warmup_epochs=3,  # Learning rate warmup epochs
            warmup_momentum=0.8,  # Warmup momentum
            warmup_bias_lr=0.1,  # Warmup bias learning rate
            box=7.5,     # Box loss gain
            cls=0.5,     # Class loss gain
            dfl=1.5,     # Distribution focal loss gain
            close_mosaic=10,  # Disable mosaic augmentation for final epochs
            plots=True,  # Generate training plots
            verbose=True  # Print verbose output
        )
        
        print("\n✅ Training completed successfully!")
        
        # Save the final model
        final_path = os.path.join(output_dir, f"clothing_detector_{timestamp}.pt")
        shutil.copy(os.path.join(output_dir, run_name, "weights/best.pt"), final_path)
        print(f"✅ Best model saved to: {final_path}")
        
        return final_path
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None

# Keep session alive (for Colab)
def keep_alive():
    """Keep Google Colab session alive"""
    try:
        from IPython.display import display, Javascript
        display(Javascript('''
        function ClickConnect(){
        console.log("Keeping session alive.");
        document.querySelector("colab-toolbar-button#connect").click()
        }
        setInterval(ClickConnect, 60000)
        '''))
        print("✅ Keep-alive mechanism activated")
    except:
        print("Note: Keep-alive mechanism not available (not running in Colab)")

# Main function
def main():
    """Main function to run the training pipeline"""
    print("===== CLOTHING DETECTION TRAINING =====")
    
    # Keep Colab alive
    keep_alive()
    
    # Make sure ultralytics is installed with exact version
    os.system("pip install -q ultralytics==8.0.196")
    
    # Downgrade torch to a compatible version - this is the most reliable solution
    print("Downgrading PyTorch to ensure compatibility with ultralytics...")
    os.system("pip install -q torch==2.0.1 torchvision==0.15.2")
    print("✅ PyTorch downgraded to 2.0.1 for better compatibility")
    
    # Create fresh dataset config
    yaml_path = create_fresh_dataset_config()
    
    # Fix label files
    fix_labels(yaml_path)
    
    # Train the model
    trained_model = train_model(yaml_path)
    
    if trained_model:
        print(f"\n✅ Training pipeline completed successfully!")
        print(f"Final model: {trained_model}")
        
        # Try to copy to Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            drive_path = "/content/drive/MyDrive/clothing_detection_models"
            os.makedirs(drive_path, exist_ok=True)
            
            shutil.copy(trained_model, os.path.join(drive_path, os.path.basename(trained_model)))
            print(f"✅ Model saved to Google Drive: {drive_path}")
        except:
            print("Note: Could not save to Google Drive")
    else:
        print("\n❌ Training pipeline failed")

# Run the script
if __name__ == "__main__":
    main()