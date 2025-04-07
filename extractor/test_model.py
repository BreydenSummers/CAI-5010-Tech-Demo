import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader
from collections import defaultdict

# Import your dataset class
class BoundingBoxDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_dir, class_id=1, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.class_id = class_id
        self.transform = transform
        
        # Get list of image files (only .jpg)
        self.imgs = [f for f in sorted(os.listdir(img_dir)) if f.endswith('.jpg')]
        
        # Restrict size of test dataset
        if len(self.imgs) > 150:
            self.imgs = self.imgs[:150]
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Get image id from filename (without extension)
        img_id = os.path.splitext(self.imgs[idx])[0]
        
        # Construct annotation file path
        ann_path = os.path.join(self.ann_dir, f"{img_id}.txt")
        
        boxes = []
        labels = []
        
        # Check if annotation file exists
        if os.path.exists(ann_path):
            # Read annotation file
            with open(ann_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                values = line.strip().split()
                if len(values) == 5:
                    try:
                        zero, x_cen, y_cen, width, height = map(float, values)
                        x_min = (x_cen - (width/2))*4800
                        x_max = (x_cen + (width/2))*4800
                        y_min = (y_cen - (height/2))*2703
                        y_max = (y_cen + (height/2))*2703
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(self.class_id)
                    except ValueError:
                        print(f"Warning: Could not parse box coordinates in {ann_path}, line: {line}")
        
        # Handle case with no annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "file_name": self.imgs[idx]  # Add filename for reference
            }
            return F.to_tensor(img), target
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "file_name": self.imgs[idx]  # Add filename for reference
        }
        
        # Apply transformations
        if self.transform:
            img, target = self.transform(img, target)
        
        # Convert image to tensor
        img = F.to_tensor(img)
        
        return img, target

def get_model(num_classes):
    # Match the model creation from training
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def evaluate_model(model, data_loader, device, iou_threshold=0.5, confidence_threshold=0.5):
    """
    Evaluate model on dataset
    """
    model.eval()
    
    results = {
        'predictions': [],
        'ground_truth': [],
        'scores': [],
        'ious': [],
        'matched_detections': 0,
        'total_ground_truth': 0,
        'total_predictions': 0,
        'file_names': []
    }
    
    per_image_results = {}
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # Run model
            outputs = model(images)
            
            # Process batch results
            for i, (output, target) in enumerate(zip(outputs, targets)):
                file_name = target['file_name']
                results['file_names'].append(file_name)
                
                # Get predicted boxes and scores above threshold
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                
                # Filter by confidence
                mask = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]
                
                # Get ground truth boxes
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                # Track predictions and ground truth
                results['predictions'].append(pred_boxes)
                results['ground_truth'].append(gt_boxes)
                results['scores'].append(pred_scores)
                
                # Initialize per-image stats
                per_image_results[file_name] = {
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'ious': [],
                    'predictions': pred_boxes.tolist(),
                    'ground_truth': gt_boxes.tolist(),
                    'scores': pred_scores.tolist()
                }
                
                # Calculate IoU for each prediction with each ground truth box
                matched_gt = set()
                
                for p_idx, pred_box in enumerate(pred_boxes):
                    max_iou = 0
                    max_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                            
                        # Only compare boxes with same class
                        if pred_labels[p_idx] == gt_labels[gt_idx]:
                            iou = calculate_iou(pred_box, gt_box)
                            if iou > max_iou:
                                max_iou = iou
                                max_gt_idx = gt_idx
                    
                    results['ious'].append(max_iou)
                    per_image_results[file_name]['ious'].append(max_iou)
                    
                    # If IoU exceeds threshold, it's a match
                    if max_iou >= iou_threshold and max_gt_idx >= 0:
                        matched_gt.add(max_gt_idx)
                        results['matched_detections'] += 1
                        per_image_results[file_name]['true_positives'] += 1
                    else:
                        per_image_results[file_name]['false_positives'] += 1
                
                # Count unmatched ground truth boxes as false negatives
                per_image_results[file_name]['false_negatives'] = len(gt_boxes) - len(matched_gt)
                
                # Update total counts
                results['total_ground_truth'] += len(gt_boxes)
                results['total_predictions'] += len(pred_boxes)
    
    # Calculate overall metrics
    results['precision'] = results['matched_detections'] / results['total_predictions'] if results['total_predictions'] > 0 else 0
    results['recall'] = results['matched_detections'] / results['total_ground_truth'] if results['total_ground_truth'] > 0 else 0
    results['f1_score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall']) if (results['precision'] + results['recall']) > 0 else 0
    results['per_image'] = per_image_results
    
    return results

def visualize_predictions(model, dataset, device, num_samples=5, confidence_threshold=0.5, output_dir='evaluation_results'):
    """
    Visualize model predictions on a sample of images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    
    # Select random samples
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    for idx in indices:
        # Get image and target
        image, target = dataset[idx]
        file_name = target['file_name']
        
        # Get model prediction
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(image_tensor)[0]
        
        # Filter predictions by confidence
        mask = prediction['scores'] >= confidence_threshold
        boxes = prediction['boxes'][mask].cpu().numpy()
        scores = prediction['scores'][mask].cpu().numpy()
        labels = prediction['labels'][mask].cpu().numpy()
        
        # Get ground truth boxes
        gt_boxes = target['boxes'].cpu().numpy()
        
        # Convert image for visualization
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = img_np * 255
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_np.astype(int))
        
        # Plot ground truth boxes
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                             fill=False, color='green', linewidth=2, label='Ground Truth')
            )
        
        # Plot predicted boxes
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                             fill=False, color='red', linewidth=2)
            )
            plt.text(x1, y1 - 5, f"Score: {score:.2f}", 
                    bbox=dict(facecolor='red', alpha=0.5), color='white', fontsize=8)
        
        plt.title(f"File: {file_name}\nGreen: Ground Truth, Red: Predictions")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"eval_{file_name}"))
        plt.close()

def main():
    # Parameters
    model_path = 'extractor/models/test_model.pth'
    img_dir = 'extractor/data/val/'  # Update with your paths
    ann_dir = 'extractor/data/val/'
    num_classes = 2  # Background + your class
    batch_size = 4
    iou_threshold = 0.5
    confidence_threshold = 0.5
    num_visualization_samples = 10
    output_dir = 'extractor/evaluation_results'
    create_visualization = False
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = BoundingBoxDataset(img_dir=img_dir, ann_dir=ann_dir, class_id=1)
    print(f"Dataset loaded with {len(dataset)} images")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    print("Loading model...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, data_loader, device, 
                             iou_threshold=iou_threshold, 
                             confidence_threshold=confidence_threshold)
    
    # Print summary results
    print("\n--- Evaluation Results ---")
    print(f"Total Images: {len(dataset)}")
    print(f"Total Ground Truth Objects: {results['total_ground_truth']}")
    print(f"Total Predictions: {results['total_predictions']}")
    print(f"Matched Detections: {results['matched_detections']}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    visualize_predictions(model, dataset, device, 
                         num_samples=num_visualization_samples,
                         confidence_threshold=confidence_threshold,
                         output_dir=output_dir)
    
    # Save results to JSON
    print(f"Saving detailed results to {output_dir}/evaluation_results.json")
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        # Create a serializable version of results
        json_results = {
            'summary': {
                'total_images': len(dataset),
                'total_ground_truth': int(results['total_ground_truth']),
                'total_predictions': int(results['total_predictions']),
                'matched_detections': int(results['matched_detections']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            },
            'per_image': {}
        }
        
        # Add per-image results
        for file_name, img_result in results['per_image'].items():
            json_results['per_image'][file_name] = {
                'true_positives': int(img_result['true_positives']),
                'false_positives': int(img_result['false_positives']),
                'false_negatives': int(img_result['false_negatives']),
                'ious': [float(iou) for iou in img_result['ious']],
                'precision': float(img_result['true_positives'] / (img_result['true_positives'] + img_result['false_positives'])) if (img_result['true_positives'] + img_result['false_positives']) > 0 else 0,
                'recall': float(img_result['true_positives'] / (img_result['true_positives'] + img_result['false_negatives'])) if (img_result['true_positives'] + img_result['false_negatives']) > 0 else 0
            }
        
        json.dump(json_results, f, indent=2)
    if create_visualization:
        # Generate confidence threshold analysis
        confidence_thresholds = np.arange(0.05, 1.0, 0.05)
        precision_values = []
        recall_values = []
        
        print("\nAnalyzing model performance across confidence thresholds...")
        for threshold in confidence_thresholds:
            threshold_results = evaluate_model(model, data_loader, device, 
                                            iou_threshold=iou_threshold, 
                                            confidence_threshold=threshold)
            precision_values.append(threshold_results['precision'])
            recall_values.append(threshold_results['recall'])
        
        # Create precision-recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall_values, precision_values, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        
        # Create confidence threshold analysis plot
        plt.figure(figsize=(10, 6))
        plt.plot(confidence_thresholds, precision_values, 'r-', label='Precision')
        plt.plot(confidence_thresholds, recall_values, 'b-', label='Recall')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Value')
        plt.title('Precision and Recall vs. Confidence Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()