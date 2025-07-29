from flask import Flask, request, jsonify, render_template
import os
from inference import load_model, predict_image

# Initialize Flask app
app = Flask(__name__)

# Load model once
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ['cat', 'dog', 'human', 'something else']
model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # templates/index.html must exist

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_path = "temp_image.jpg"
    file.save(temp_path)

    try:
        label, probs = predict_image(temp_path, model, CLASS_NAMES)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Convert probabilities to percentages
    prob_dict = {CLASS_NAMES[i]: float(probs[i].item()) * 100 for i in range(len(CLASS_NAMES))}
    
    # Format text
    output_text = "Prediction results:\n"
    for cls, prob in prob_dict.items():
        output_text += f"{cls}: {prob:.2f}%\n"
    output_text += f"\nThe image is probably a {label}."
    
    return jsonify({'output_text': output_text})


if __name__ == "__main__":
    app.run(debug=True)
