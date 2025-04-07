import torch
import timm
from torchvision import transforms
from PIL import Image
import sys
import os

# Check if an image path was provided
if len(sys.argv) < 2:
    print("Usage: python inference.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    sys.exit(1)

# Use MPS if available, otherwise CUDA or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# Same transform as used in validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("best_retina_model.pth", map_location=device))
model.to(device)
model.eval()

# Class names (must match the order used during training)
class_names = ['Abnormal', 'Normal']

# Load and preprocess the image
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

# Get prediction
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    predicted_class_idx = outputs.argmax(dim=1).item()
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item() * 100

print(f"\nImage: {image_path}")
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

# Print probabilities for each class
print("\nClass probabilities:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {probabilities[i].item()*100:.2f}%") 