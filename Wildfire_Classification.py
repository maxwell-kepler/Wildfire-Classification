import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import PIL
from PIL import ImageFile

import os
from datetime import datetime
import traceback




class PretrainedWildfireNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PretrainedWildfireNet, self).__init__()
        # Load the pretrained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters by default
        for param in self.model.parameters():
            param.requires_grad = False
                    
        # Replace the final fully connected layer with the custom classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)





def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, patience=3):
    start_time = datetime.now()
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad() # Zero the parameter gradients
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train': # Backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Early stopping check
            if phase == 'val':
                scheduler.step(epoch_loss) # Step the learning rate scheduler based on validation loss
                
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    early_stopping_counter = 0           
                    torch.save(model.state_dict(), 'best_wildfire_model.pth') # Save the model
                    print("Model saved")
                else:
                    early_stopping_counter += 1
                    print(f"Early stopping counter: {early_stopping_counter} / {patience}")            
                    if early_stopping_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        end_time = datetime.now()
                        time_elapsed = end_time - start_time
                        print(f'Training complete in {time_elapsed}')
                        return model
    
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    print(f'Training complete in {time_elapsed}')
    return model

# Unfreeze layer4 for finetuning model
def unfreeze_layer4(model):
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
    print("Unfroze layer4 for fine-tuning")

def get_optimizer_finetune(model):
    # Lower learning rate for pretrained layer4
    return optim.Adam([
        {"params": model.model.layer4.parameters(), "lr": 1e-5},
        {"params": model.model.fc.parameters(), "lr": 1e-4}
    ], weight_decay=1e-5)


# Set up paths to data directories
data_dir = "/home/maxwell.kepler/TransferLearning/FinalProject/Data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

# Data transformations for training and evaluation
# The images are all 350x350, but the pre-trained model expect 224x224 images so they've been resized
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
    ]),
    "eval": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet statistics
    ])
}

# Sometimes there was an error given that an image was corrupted, but I couldn't find it
# Custom image loader that can handle corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # This is needed to handle truncated images
def safe_PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            img = PIL.Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"Warning: Error loading image {path}: {e}")
        return PIL.Image.new('RGB', (224, 224), (0, 0, 0)) # Return a black image of the expected size

# Custom loader to avoid corruption
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(SafeImageFolder, self).__init__(root=root, transform=transform, loader=safe_PIL_loader)

# Dataset objects from the SafeImageFolder class
datasets_dict = {
    "train": SafeImageFolder(train_dir, transform=transform["train"]),
    "val": SafeImageFolder(val_dir, transform=transform["eval"]),
    "test": SafeImageFolder(test_dir, transform=transform["eval"])
}

# Class information
print(f"Classes: {datasets_dict['train'].classes}")
print(f"Class after mapping: {datasets_dict['train'].class_to_idx}")

# Create DataLoaders
num_workers = 2 # Reduce num_workers to avoid warnings and potential freezing
batch_size = 64
dataloaders = {
    "train": DataLoader(datasets_dict["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
    "val": DataLoader(datasets_dict["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    "test": DataLoader(datasets_dict["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
}
print(f"Training set size: {len(datasets_dict['train'])}")
print(f"Validation set size: {len(datasets_dict['val'])}")
print(f"Test set size: {len(datasets_dict['test'])}")

# Set up device, using the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = PretrainedWildfireNet(num_classes=len(datasets_dict['train'].classes)).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Create optimizer with proper parameter groups and learning rates
optimizer = optim.Adam(model.model.fc.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler: Reduce LR on Plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Test the data loading process
def check_data_loading():
    print("\nChecking data loading")
    # Check a batch from each dataloader
    for phase in ['train', 'val', 'test']:
        try:
            inputs, labels = next(iter(dataloaders[phase]))
            print(f"{phase} batch shape: {inputs.shape}")
            print(f"{phase} labels: {labels[:10]}")
            if torch.isnan(inputs).any():
                print(f"WARNING: NaN values detected in {phase} batch")
        except Exception as e:
            print(f"Error loading {phase} batch: {e}")
    print("Data loading check complete")
check_data_loading()


def test_model(model, test_loader):
    model.eval()
    running_corrects = 0
    
    print("\nEvaluating on test set")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')

# Train and evaluate model
try:
    # ---- PHASE 1: Train classifier only ----
    for param in model.parameters():
        param.requires_grad = False  # freeze all
    for param in model.model.fc.parameters():
        param.requires_grad = True  # unfreeze classifier
        
    print("\nStarting PHASE 1: Train classifier head only")
    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, patience=3)
    torch.save(trained_model.state_dict(), "best_before_fine_tuned_model.pth")


    # ---- PHASE 2: Unfreeze layer4 (last layer before fc) to learn more features and fine-tune ----
    print("\nStarting PHASE 2: Fine-tune layer4")
    model.load_state_dict(torch.load("best_wildfire_model.pth"))
    unfreeze_layer4(model)

    optimizer = get_optimizer_finetune(model)
    # Switching learning rate scheduler: Cosine Annealing 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5, patience=2)
    torch.save(trained_model.state_dict(), "fine_tuned_model.pth")

    # Evaluate final model
    model.load_state_dict(torch.load("fine_tuned_model.pth"))
    test_model(model, dataloaders['test'])

except Exception as e:
    print(f"Error during training or testing: {e}")
    traceback.print_exc()
