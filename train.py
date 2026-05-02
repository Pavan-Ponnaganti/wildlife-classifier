import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm
from torch.utils.data import DataLoader
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import csv
import pandas as pd

class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training.")

    # Data paths - adjust based on where the dataset is actually extracted
    data_dir = 'dataset/animals/animals'
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}. Please place it there.")
        return

    # Transforms (using standard ImageNet normalization)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset loading
    full_dataset = datasets.ImageFolder(data_dir)
    
    # Splitting dataset into train/val/test (70/15/15)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    train_data = TransformWrapper(train_dataset, transform=train_transforms)
    val_data = TransformWrapper(val_dataset, transform=val_transforms)
    test_data = TransformWrapper(test_dataset, transform=val_transforms)

    # Save class names for inference
    class_names = full_dataset.classes
    with open('classes.json', 'w') as f:
        json.dump(class_names, f)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    # Setup pre-trained model (EfficientNet-B0) using timm
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(class_names))
    
    # We freeze the base model initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Read unfreeze level from environment variable (default 1)
    unfreeze_level = int(os.environ.get("UNFREEZE_LEVEL", "1"))
    
    # LEVEL 0: Classifier Only
    for param in model.get_classifier().parameters():
        param.requires_grad = True
        
    # LEVEL 1: Unfreeze top conv block
    if unfreeze_level >= 1:
        if hasattr(model, 'conv_head'):
            for param in model.conv_head.parameters():
                param.requires_grad = True
        if hasattr(model, 'bn2'):
            for param in model.bn2.parameters():
                param.requires_grad = True
                
    # LEVEL 2: Unfreeze block 6 (the very last inverted residual block)
    if unfreeze_level >= 2:
        if hasattr(model, 'blocks'):
            for param in model.blocks[-1].parameters():
                param.requires_grad = True
                
    # LEVEL 3: Unfreeze block 5
    if unfreeze_level >= 3:
        if hasattr(model, 'blocks') and len(model.blocks) >= 2:
            for param in model.blocks[-2].parameters():
                param.requires_grad = True
                
    # LEVEL 4: Unfreeze block 4
    if unfreeze_level >= 4:
        if hasattr(model, 'blocks') and len(model.blocks) >= 3:
            for param in model.blocks[-3].parameters():
                param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Update optimizer to include all unfrozen parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Print parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (Unfrozen): {trainable_params:,}")

    epochs = 5 # Light fine-tuning
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'efficientnet_b0_animals.pth')
    print("Training complete and model saved as efficientnet_b0_animals.pth")

    # Generate Confusion Matrix on Test Set
    print("Evaluating on test set to generate Confusion Matrix...")
    model.eval()
    all_preds = []
    all_labels = []
    total_test = 0
    correct_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            total_test += labels.size(0)
            correct_test += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Final Test Accuracy: {correct_test / total_test:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate TP, TN, FP, FN per class
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)
    
    with open('confusion_matrix_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'TP', 'TN', 'FP', 'FN'])
        for i in range(len(class_names)):
            writer.writerow([class_names[i], tp[i], tn[i], fp[i], fn[i]])
    print("Class-wise TP, TN, FP, FN saved as 'confusion_matrix_metrics.csv'!")

    # Calculate Global (Mixed) Metrics
    global_tp = tp.sum()
    global_tn = tn.sum()
    global_fp = fp.sum()
    global_fn = fn.sum()
    print(f"\n--- Global Model Metrics (All Classes Combined) ---")
    print(f"Total True Positives (TP): {global_tp}")
    print(f"Total True Negatives (TN): {global_tn}")
    print(f"Total False Positives (FP): {global_fp}")
    print(f"Total False Negatives (FN): {global_fn}")
    print("---------------------------------------------------\n")

    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix saved as 'confusion_matrix.png'!")

    # Live Ablation Graph Generation
    test_acc_percent = (correct_test / total_test) * 100
    params_in_millions = trainable_params / 1e6
    
    ablation_file = 'ablation_results.csv'
    new_data = pd.DataFrame({'Trainable_Params_Millions': [params_in_millions], 'Test_Accuracy': [test_acc_percent]})
    
    if not os.path.exists(ablation_file):
        new_data.to_csv(ablation_file, index=False)
    else:
        new_data.to_csv(ablation_file, mode='a', header=False, index=False)
        
    print(f"Added {params_in_millions:.2f}M params / {test_acc_percent:.2f}% accuracy to ablation_results.csv")
    
    # Read and plot the live data
    all_runs = pd.read_csv(ablation_file)
    # If they ran the same config twice, take the best accuracy for that parameter count
    all_runs = all_runs.groupby('Trainable_Params_Millions').max().reset_index()
    
    if len(all_runs) > 1:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(data=all_runs, x='Trainable_Params_Millions', y='Test_Accuracy', marker='o', markersize=10, linewidth=2.5, color='#1f77b4')
        
        for i in range(len(all_runs)):
            ax.annotate(f"{all_runs['Test_Accuracy'].iloc[i]:.2f}%", 
                        (all_runs['Trainable_Params_Millions'].iloc[i], all_runs['Test_Accuracy'].iloc[i]),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9))
                        
        plt.title('Live Ablation Study: Trainable Parameters vs. Model Accuracy', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Number of Trainable Parameters (Millions)', fontsize=12, fontweight='bold')
        plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        
        y_min = all_runs['Test_Accuracy'].min() - 2
        y_max = all_runs['Test_Accuracy'].max() + 2
        plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig('live_ablation_study_graph.png', dpi=300)
        print("Live Ablation Graph updated and saved as 'live_ablation_study_graph.png'!")

if __name__ == "__main__":
    main()
