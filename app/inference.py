import torch
from PIL import Image
import torchvision.transforms as transforms
from app.cnn_model import EnhancedCNN, CLASSES
import os

device = torch.device("cpu")

# Load the trained model
model = EnhancedCNN().to(device)
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()
    return {
        "class": CLASSES[pred],
        "confidence": round(confidence, 4)
    }