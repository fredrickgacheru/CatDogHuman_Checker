import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model (only once)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.conf = 0.5  # confidence threshold
LABEL_MAP = {'cat': 15, 'dog': 16, 'human': 0}  # COCO IDs

# ------------------------
# Helper Functions
# ------------------------

def preprocess_with_yolo(image_path, resize=(224, 224)):
    """Detects cat/dog/human and crops image using YOLOv5."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    results = yolo_model(image)
    detections = results.xyxy[0]

    for *box, conf, cls in detections:
        if int(cls) in LABEL_MAP.values():
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            resized = cv2.resize(cropped, resize)
            return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    return None  # no relevant object detected

def load_model(model_path, num_classes=4):
    """Loads the trained ResNet50 model."""
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, class_names):
    """Preprocess image, run YOLO + ResNet, and return label + probabilities."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # must match training normalization
    ])

    # YOLO preprocessing
    preprocessed_image = preprocess_with_yolo(image_path)
    if preprocessed_image is None:
        return "something else", None

    # Transform for ResNet
    image_tensor = transform(preprocessed_image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probs, 0)

    predicted_label = class_names[top_class.item()]
    return predicted_label, probs.cpu().numpy()

# ------------------------
# Optional: Standalone Test
# ------------------------
if __name__ == "__main__":
    model = load_model("best_model.pth", num_classes=4)
    class_names = ['cat', 'dog', 'human', 'something else']
    label, probs = predict_image("demo_images/dog_1.jpg", model, class_names)
    print("Predicted:", label)
    if probs is not None:
        print("Probabilities:", probs)
