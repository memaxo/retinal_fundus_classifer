import torch
from torch import nn
from torchvision import datasets, transforms
import timm
from torchmetrics import Accuracy, F1Score
import os
from torch.utils.data import DataLoader, random_split

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

train_dir = "train"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),              # resize images to 224x224 pixels
    transforms.RandomHorizontalFlip(p=0.5),     # random horizontal flip for augmentation
    transforms.ToTensor(),                      # convert PIL image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], # normalize to ImageNet mean
                         [0.229, 0.224, 0.225]) # normalize to ImageNet std
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


val_dataset.dataset = datasets.ImageFolder(train_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
class_names = full_dataset.classes
print("Classes:", class_names)
print(f"Training on {train_size} samples, validating on {val_size} samples")
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10  # adjusted for small dataset

best_f1 = 0.0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()               # reset gradients from previous step
        outputs = model(images)             # forward pass
        loss = criterion(outputs, labels)   # compute loss
        loss.backward()                     # backpropagate gradients
        optimizer.step()                    # update model parameters
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    
    model.eval()
    acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=2, average="macro").to(device)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
    
    accuracy = acc_metric.compute().item()
    f1_score = f1_metric.compute().item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, "
          f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")
    
    if f1_score > best_f1:
        best_f1 = f1_score
        torch.save(model.state_dict(), "best_retina_model.pth")
        print(f"Saved new best model with F1 score: {f1_score:.4f}")

print("Training complete!")

model.load_state_dict(torch.load("best_retina_model.pth"))
model.eval()

final_acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
final_f1_metric = F1Score(task="multiclass", num_classes=2, average="macro").to(device)

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        final_acc_metric.update(preds, labels)
        final_f1_metric.update(preds, labels)

final_accuracy = final_acc_metric.compute().item()
final_f1_score = final_f1_metric.compute().item()

print(f"Final Model - Validation Accuracy: {final_accuracy:.4f}, F1 Score: {final_f1_score:.4f}") 