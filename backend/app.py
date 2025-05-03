from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
from datetime import datetime
import requests
from PIL import Image
import io
import base64
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"  # OpenRouter API key
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# WM API Configuration
WM_API_URL = "https://apitest.wm.com/v1"

def analyze_with_mistral(results):
    """Analyze detection results using Mistral-7B through OpenRouter"""
    
    # Create summary from detection results only
    detection_summary = []
    waste_types = set()
    
    # Collect unique waste types
    for det in results["detections"]:
        waste_types.add(det['class_name'].lower())
    
    # Create a summary of detected waste types
    for waste_type in waste_types:
        detection_summary.append(f"- Detected waste type: {waste_type}")
    
    prompt = f"""Based on the following waste detection results, provide recommendations:

Detection Results:
{chr(10).join(detection_summary)}

Please provide:
1. Environmental impact analysis
2. Proper disposal recommendations
3. Safety precautions
4. Recycling guidelines
"""

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:5000",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status()
        
        analysis = response.json()["choices"][0]["message"]["content"]
        return {
            "success": True,
            "analysis": analysis,
            "detected_waste_types": list(waste_types)
        }
        
    except Exception as e:
        print(f"Error calling Mistral-7B: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
# Initialize models as None
waste_model = None
medwaste_model = None
ewaste_model = None

# Create results directory if it doesn't exist
RESULTS_DIR = 'detection_results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Define class names for waste model
WASTE_CLASS_NAMES = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "metal",
    4: "trash",
    5: "organic",
    6: "plastic"
}

# Define class names for medical waste model
MEDWASTE_CLASS_NAMES = {
    0: "biohazard",
    1: "medical_waste",
    2: "needle",
    3: "syringe"
}

# Define class names for batteries model
EWASTE_CLASS_NAMES = {
    0: "Baterai 9v",
    1: "Baterai AA",
    2: "Battery",
    3: "Battery-Hazardous",
    4: "battery",
    5: "drycell"
}

def save_detection_results(image_name, detections):
    """Save detection results to a JSON file and analyze with Mistral"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Prepare results with additional metadata
    results = {
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "detections": detections,
        "summary": {
            "waste_detections": len([d for d in detections if d['model'] == 'waste']),
            "medical_waste_detections": len([d for d in detections if d['model'] == 'medwaste']),
            "battery_detections": len([d for d in detections if d['model'] == 'ewaste'])
        }
    }
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Get analysis from Mistral-7B
    analysis_result = analyze_with_mistral(results)
    
    # Add analysis to results and save updated JSON
    results["ai_analysis"] = analysis_result
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return filename, analysis_result

import os
from ultralytics import YOLO

def load_models():
    global waste_model, medwaste_model, ewaste_model
    try:
        # Get the absolute path to the root of the project
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Construct model file paths relative to project root
        WASTE_MODEL_PATH = os.path.join(project_root, 'models', 'best.pt')
        MEDWASTE_MODEL_PATH = os.path.join(project_root, 'models', 'medwaste.pt')
        EWASTE_MODEL_PATH = os.path.join(project_root, 'models', 'batteeries.pt')  # check spelling

        # Load waste model
        if not os.path.exists(WASTE_MODEL_PATH):
            raise FileNotFoundError(f"Waste model file not found at {WASTE_MODEL_PATH}")
        waste_model = YOLO(WASTE_MODEL_PATH)
        print("Waste model loaded successfully!")

        # Load medical waste model
        if not os.path.exists(MEDWASTE_MODEL_PATH):
            raise FileNotFoundError(f"Medical waste model file not found at {MEDWASTE_MODEL_PATH}")
        medwaste_model = YOLO(MEDWASTE_MODEL_PATH)
        print("Medical waste model loaded successfully!")

        # Load batteries model
        if not os.path.exists(EWASTE_MODEL_PATH):
            raise FileNotFoundError(f"Batteries model file not found at {EWASTE_MODEL_PATH}")
        ewaste_model = YOLO(EWASTE_MODEL_PATH)
        print("Batteries model loaded successfully!")

        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    if waste_model is None or medwaste_model is None or ewaste_model is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Models not loaded"
        }), 500
    return jsonify({
        "status": "healthy",
        "message": "Backend is running and models are loaded"
    })

@app.route('/api/process-image', methods=['POST'])
def process_image():
    if waste_model is None or medwaste_model is None or ewaste_model is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please check server logs.'
        }), 500

    try:
        # Get the image file from the request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        file = request.files['image']
        image_name = file.filename
        
        # Read the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process the image with all models
        waste_results = waste_model(image)
        medwaste_results = medwaste_model(image)
        
        # Process batteries model with confidence threshold
        ewaste_results = ewaste_model(image, conf=0.25)  # Lower confidence threshold
        
        # Get the first result from each model
        waste_result = waste_results[0]
        medwaste_result = medwaste_results[0]
        ewaste_result = ewaste_results[0]
        
        # Debug batteries results
        print("Batteries model results:")
        print(f"Number of detections: {len(ewaste_result.boxes)}")
        if len(ewaste_result.boxes) > 0:
            print("Battery detections:")
            for box in ewaste_result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"Class: {class_id} ({EWASTE_CLASS_NAMES.get(class_id, 'Unknown')}), Confidence: {conf}")
        
        # Combine the results
        combined_image = waste_result.plot()
        combined_image = Image.fromarray(combined_image)
        
        # Convert the processed image to base64
        buffered = io.BytesIO()
        combined_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Get detection results from all models
        detections = []
        
        # Add waste detections
        for box in waste_result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                'model': 'waste',
                'class': class_id,
                'class_name': WASTE_CLASS_NAMES.get(class_id, f"Unknown ({class_id})"),
                'confidence': float(box.conf[0]),
                'box': box.xyxy[0].tolist()
            })
        
        # Add medical waste detections
        for box in medwaste_result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                'model': 'medwaste',
                'class': class_id,
                'class_name': MEDWASTE_CLASS_NAMES.get(class_id, f"Unknown ({class_id})"),
                'confidence': float(box.conf[0]),
                'box': box.xyxy[0].tolist()
            })
        
        # Add battery detections
        for box in ewaste_result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                'model': 'ewaste',
                'class': class_id,
                'class_name': EWASTE_CLASS_NAMES.get(class_id, f"Unknown ({class_id})"),
                'confidence': float(box.conf[0]),
                'box': box.xyxy[0].tolist()
            })

        # Save results and get AI analysis
        json_filename, analysis_result = save_detection_results(image_name, detections)
        
        return jsonify({
            'success': True,
            'processed_image': img_str,
            'detections': detections,
            'json_file': json_filename,
            'ai_analysis': analysis_result
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    if load_models():
        app.run(debug=True, port=5000)
    else:
        print("Failed to start server: Models could not be loaded") 