import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import random

class BoundingBoxDataset(Dataset):
    def __init__(self, img_dir, ann_dir, class_id=1, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images (.jpg format)
            ann_dir (string): Directory with all the annotation text files
            class_id (int): Class ID to assign to all bounding boxes (default: 1)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.class_id = class_id
        self.transform = transform
        
        # Get list of image files (only .jpg)
        self.imgs = [f for f in sorted(os.listdir(img_dir)) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Get image id from filename (without extension)
        img_id = os.path.splitext(self.imgs[idx])[0]
        
        # Construct annotation file path (assuming same name with .txt extension)
        ann_path = os.path.join(self.ann_dir, f"{img_id}.txt")
        
        boxes = []
        labels = []
        
        # Check if annotation file exists
        if os.path.exists(ann_path):
            # Read annotation file
            with open(ann_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                # Parse line containing "x_min y_min x_max y_max"
                values = line.strip().split()
                if len(values) == 5:
                    try:
                        # Convert to float
                        zero, x_cen, y_cen, width, height = map(float, values)
                        x_min = (x_cen - (width/2))*4800
                        x_max = (x_cen + (width/2))*4800
                        y_min = (y_cen - (height/2))*2703
                        y_max = (y_cen + (height/2))*2703
                        boxes.append([x_min, y_min, x_max, y_max])
                        # Use the provided class_id for all boxes
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
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
            return F.to_tensor(img), target
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        # Calculate area using PyTorch operations
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Apply transformations
        if self.transform:
            img, target = self.transform(img, target)
        
        # Convert image to tensor
        img = F.to_tensor(img)
        
        return img, target

def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    
    running_loss = 0.0
    i = 0
    
    for images, targets in data_loader:
        i += 1
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass and optimize
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        
        # Print batch loss for tracking progress
        if i % 10 == 0:
            print(f"  Batch {i}, Loss: {losses.item():.4f}")
    
    return running_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    
    # Simple validation function - just run forward pass
    val_loss = 0.0
    i = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            i += 1
            # Move images and targets to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Just run inference to check for errors
            try:
                outputs = model(images)
                print(f"  Validation batch {i}: OK")
            except Exception as e:
                print(f"  Validation batch {i} failed: {str(e)}")
    
    return

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_sample(image, target):
    """
    Visualize a sample image with its ground truth bounding boxes
    """
    # Convert tensor to numpy for visualization
    img_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Scale pixel values if needed
    if img_np.max() <= 1.0:
        img_np = img_np * 255
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np.astype(int))
    
    # Get bounding boxes
    boxes = target["boxes"].cpu().numpy()
    labels = target["labels"].cpu().numpy()
    
    # Plot each box
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         fill=False, color='red', linewidth=2)
        )
        plt.text(x1, y1, f"Class: {label}", 
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.savefig('extractor/samples/sample_with_boxes.png')
    print("Sample image with bounding boxes saved as 'extractor/sample/sample_with_boxes.pngs'")

def main():
    # Set device (CPU)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Paths
    img_dir = 'extractor/data/test/'  # Replace with actual path
    ann_dir = 'extractor/data/test/'  # Replace with actual path
    
    # Dataset parameters
    num_classes = 2  # Background + your class
    batch_size = 2
    
    print("Creating dataset...")
    try:
        # Create dataset and data loaders
        dataset = BoundingBoxDataset(img_dir=img_dir, ann_dir=ann_dir, class_id=1)
        print(f"Dataset created successfully with {len(dataset)} images")
        
        # Visualize a sample to verify data loading
        if len(dataset) > 0:
            sample_img, sample_target = dataset[0]
            print(f"Sample image shape: {sample_img.shape}")
            print(f"Sample target boxes: {sample_target['boxes'].shape}")
            # Save sample visualization
            visualize_sample(sample_img, sample_target)
        
        # Split dataset into train and test
        indices = list(range(len(dataset)))
        split = int(len(dataset) * 0.8)
        
        # Randomize indices
        random.shuffle(indices)
        
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print("Creating model...")
        # Get the model
        model = get_model(num_classes)
        model.to(device)
        
        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Number of epochs
        num_epochs = 7
        
        print("Starting training...")
        # Training loop
        try:
            for epoch in range(num_epochs):
                start_time = time.time()
                
                print(f"Epoch {epoch+1}/{num_epochs}")
                # Train for one epoch
                train_loss = train_one_epoch(model, optimizer, train_loader, device)
                
                # Update the learning rate
                lr_scheduler.step()
                
                # Evaluate on the validation set
                print("Validating...")
                evaluate(model, val_loader, device)
                
                end_time = time.time()
                epoch_time = end_time - start_time
                
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")
        except KeyboardInterrupt as e:
            print("Interrupted... Saving model")
        # Save the model
        try:
            torch.save(model.state_dict(), 'extractor/models/test_model.pth')
            print("Training complete! Model saved as 'models/test_model.pth'")
        except RuntimeError as e:
            print(e)
            print("Saving model as model.pth in top-level directory")
            torch.save(model.state_dict(), 'model.pth')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()