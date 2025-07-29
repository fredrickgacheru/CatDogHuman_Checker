# 🐶🐱🧑 Cat-Dog-Human Image Classifier

This project is a web-based image classifier that takes an image input and predicts whether the image contains a **cat**, **dog**, **human**, or **something else**. Built using deep learning techniques with **PyTorch**, the app leverages **YOLOv5** for object detection and **ResNet-50** for classification. The web interface is built with **Flask**.

---

## 🚀 Demo

Upload an image and receive real-time predictions like:

>  Cat: 85%,🐶 Dog: 10% 🧑 Human: 4 ❓ Other: 1%

---

## 🧠 Project Objective

To create a machine learning pipeline that:
- Detects the main object in an image.
- Classifies it as one of four categories: **cat**, **dog**, **human**, or **other**.
- Provides confidence scores for each class.
- Delivers results via a user-friendly web interface.

---

## 📂 Dataset

### Sources
- **Cats and Dogs**: [Oxford Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Humans**: [LFW Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)

### Initial Image Counts
- 🐱 Cats: 2,599
- 🐶 Dogs: 4,791
- 🧑 Humans: 13,233+

After YOLOv5 filtering and curation:
- 🐱 Cats: 1,139
- 🐶 Dogs: 3,282

Balanced to 3,000 images per class (via augmentation and subsampling).

---

## 🧪 Data Preparation Pipeline

1. **YOLOv5 Object Detection** – Detect and crop objects (cats/dogs).
2. **Resize** – All images resized to `224x224`.
3. **Augment (Training only)** – Rotation, flipping, brightness, etc.
4. **Transform** – Convert to tensors, normalize to `[-1, 1]`.
5. **Split** – `80%` Train, `20%` Validation, `20%` Test.

| Step         | Train ✅ | Validation ✅ | Test ✅ |
|--------------|---------|---------------|---------|
| Resize       | ✅      | ✅            | ✅      |
| Augment      | ✅      | ❌            | ❌      |
| ToTensor     | ✅      | ✅            | ✅      |
| Normalize    | ✅      | ✅            | ✅      |

---

## 🏗️ Model Architecture

- **Base Model**: `ResNet-50` (pre-trained on ImageNet)
- **Transfer Learning**:
  - Frozen base layers
  - Fine-tuned final block (`layer4`) and classifier head
  - Final layer modified to output 4 classes
- **Loss Function**: Cross Entropy
- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau

---

## 📈 Training Details

- **Epochs**: 10
- **Batch Size**: 32
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix



---

## 🌐 Web Application

Built using **Flask**, the web app allows users to upload an image and receive predictions instantly.

### Inference Pipeline
1. Upload image
2. Apply YOLOv5 detection
3. Crop → Resize → Normalize
4. Feed into ResNet-50 model
5. Return predictions + confidence scores

## IMPORTANT NOTE
For better understanding of the topic, kindly check the "Notes on the Project.docx" file. 

# 📦 Installation

```bash
# Clone repo
git clone https://github.com/fredrickgacheru/CatDogHuman_Checker.git


# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

