import os
import torch
import json
import argparse
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image

def get_model(num_classes):
    """Load model architecture with the correct number of classes"""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def predict_image(model, image_path, device, confidence_threshold=0.5):
    """
    Run model prediction on a single image
    Returns the predictions and the processed image
    """
    # Load and process image
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    
    # Run model prediction
    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    
    # Filter predictions by confidence
    mask = prediction['scores'] >= confidence_threshold
    boxes = prediction['boxes'][mask].cpu().numpy()
    scores = prediction['scores'][mask].cpu().numpy()
    labels = prediction['labels'][mask].cpu().numpy()
    
    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }, img

def visualize_predictions(img, predictions, output_path):
    """
    Visualize predictions on the image and save to file
    """
    # Convert PIL image for matplotlib visualization
    img_np = plt.imread(img.filename) if hasattr(img, 'filename') else F.to_tensor(img).permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    
    # Plot predicted boxes
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x1, y1 - 5, f"Score: {score:.2f}", 
                bbox=dict(facecolor='red', alpha=0.5), color='white', fontsize=8)
    
    plt.title(f"Object Detection Results")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def save_predictions_to_json(predictions, image_path, output_path):
    """
    Save prediction results to JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'image_path': image_path,
        'predictions': {
            'boxes': predictions['boxes'].tolist(),
            'scores': predictions['scores'].tolist(),
            'labels': predictions['labels'].tolist(),
            'num_detections': len(predictions['scores'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Prediction results saved to {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Object Detection on a single image')
    parser.add_argument('--input', required=True, help='Path to input image file')
    parser.add_argument('--model', default='extractor/models/test_model.pth', help='Path to model file')
    parser.add_argument('--output_dir', default='extractor/output', help='Directory to save results')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = get_model(num_classes=2)  # Adjust number of classes as needed
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process image
    try:
        # Get predictions
        predictions, img = predict_image(model, args.input, device, args.confidence)
        
        # Generate output paths
        input_filename = os.path.basename(args.input)
        base_name, ext = os.path.splitext(input_filename)
        
        # Save detection image with _detection suffix
        output_image = os.path.join(args.output_dir, f"{base_name}_detection{ext}")
        
        # Save JSON with the same base name as the original image
        output_json = os.path.join(args.output_dir, f"{base_name}.json")
        
        # Save visualization
        visualize_predictions(img, predictions, output_image)
        
        # Save predictions to JSON
        save_predictions_to_json(predictions, args.input, output_json)
        
        print(f"Processing complete. Results saved to {args.output_dir}")
        print(f"Detection image: {output_image}")
        print(f"Metadata JSON: {output_json}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return

if __name__ == "__main__":
    main()