from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import json
from datetime import datetime
import requests
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import traceback
import time

from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import openai
from diffusers import StableDiffusionXLPipeline
import torch
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
CORS(app)

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# WM API Configuration
WM_API_URL = "https://apitest.wm.com/v1"

# Mistral API Configuration
openai.api_key = "1QUMWhu2cSOcEupqAvsyuU0j37aCnOqb"
openai.api_base = "https://api.mistral.ai/v1"

# Model paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WASTE_MODEL_PATH = os.path.join(project_root, 'models', 'best.pt')
MEDWASTE_MODEL_PATH = os.path.join(project_root, 'models', 'medwaste.pt')
EWASTE_MODEL_PATH = os.path.join(project_root, 'models', 'batteeries.pt')
EFFICIENT_MODEL_PATH = os.path.join(project_root, 'models', 'efficient1.h5')

# Output directory
RESULTS_DIR = 'detection_results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Class dictionaries
WASTE_CLASS_NAMES = {0: "glass", 1: "paper", 2: "cardboard", 3: "metal", 4: "trash", 5: "organic", 6: "plastic"}
MEDWASTE_CLASS_NAMES = {0: "biohazard", 1: "medical_waste", 2: "needle", 3: "syringe"}
EWASTE_CLASS_NAMES = {0: "Baterai 9v", 1: "Baterai AA", 2: "Battery", 3: "Battery-Hazardous", 4: "battery", 5: "drycell"}
CLASSIFICATION_CLASS_NAMES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 
    'cardboard_boxes', 'clothing', 'food_waste', 
    'glass_beverage_bottles', 'glass_cosmetic_containers', 
    'office_paper', 'paper_cups', 'plastic_detergent_bottles', 
    'plastic_shopping_bags', 'plastic_soda_bottles', 
    'plastic_straws', 'plastic_water_bottles'
]

# Model variables
waste_model = None
medwaste_model = None
ewaste_model = None
efficientnet_model = None
classification_model = None

# Initialize Stable Diffusion pipeline
sd_pipeline = None

# ----------- Utility functions -----------

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

def analyze_with_mistral(results):
    """Analyze detection results using Mistral-7B through OpenRouter"""
    detection_summary = []
    waste_types = set()

    for det in results["detections"]:
        waste_types.add(det['class_name'].lower())

    for waste_type in waste_types:
        detection_summary.append(f"- Detected waste type: {waste_type}")

    prompt = f"""Based on the following waste detection results, provide recommendations:

Detection Results:
{chr(10).join(detection_summary)}

Please provide:
Environmental impact analysis
Proper disposal recommendations
Safety precautions
Recycling guidelines
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

def load_models():
    """Load all required models"""
    global waste_model, medwaste_model, ewaste_model, efficientnet_model, classification_model
    try:
        # Get the absolute path to the root of the project
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Load YOLO models
        WASTE_MODEL_PATH = os.path.join(project_root, 'models', 'best.pt')
        MEDWASTE_MODEL_PATH = os.path.join(project_root, 'models', 'medwaste.pt')
        EWASTE_MODEL_PATH = os.path.join(project_root, 'models', 'batteeries.pt')
        CLASSIFICATION_MODEL_PATH = os.path.join(project_root, 'models', 'efficient1.h5')

        waste_model = YOLO(WASTE_MODEL_PATH)
        medwaste_model = YOLO(MEDWASTE_MODEL_PATH)
        ewaste_model = YOLO(EWASTE_MODEL_PATH)
        classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)

        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def classify_with_efficientnet(image_pil):
    try:
        image_resized = image_pil.resize((256, 256))
        img_array = np.array(image_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = efficientnet_model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        return CLASSIFICATION_CLASS_NAMES[class_idx], confidence
    except Exception as e:
        print(f"EfficientNet error: {e}")
        return "unknown", 0.0

def prepare_image_for_classification(img, target_size=(256, 256)):
    """Prepare image for classification model"""
    if isinstance(img, Image.Image):
        img = img.resize(target_size)
        img_array = np.array(img)
    else:
        img_array = img

    # Preprocess for EfficientNet
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def load_classification_model(model_path):
    """Load classification model"""
    try:
        # Load the model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'preprocess_input': preprocess_input
            }
        )
        print("Classification model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading classification model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def initialize_sd_pipeline():
    global sd_pipeline
    try:
        print("\n🚀 Initializing Stable Diffusion...")
        if torch.cuda.is_available():
            print("✅ Using GPU for Stable Diffusion")
            sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                local_files_only=True  # Use cached model if available
            )
            sd_pipeline = sd_pipeline.to("cuda")
        else:
            print("⚠️ Using CPU for Stable Diffusion (slower)")
            # Use a smaller model for CPU
            sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float32,
                use_safetensors=True,
                local_files_only=True,  # Use cached model if available
                low_cpu_mem_usage=True  # Enable low memory mode
            )
            sd_pipeline = sd_pipeline.to("cpu")
        print("✅ Stable Diffusion pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Stable Diffusion: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        sd_pipeline = None

def generate_description_with_mistral(pred_class):
    prompt = (
        f"You are a sustainable product designer. Propose one specific, useful and realistic object made only from recycled '{pred_class}'. "
        "Keep the description short and precise, suitable for image generation. The object must be 100% made from that material and usable in daily life. "
        "Include the object's name, what it is used for, key visual features (like size, shape, texture, and color), and how it looks on a studio background. "
        "Output in one single sentence formatted like: "
        "\"a [object] made from recycled [material], [key visual details], shown on a clean white studio background\"."
    )

    try:
        print(f"\n📝 Generating description for {pred_class}...")
        response = openai.ChatCompletion.create(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        description = response["choices"][0]["message"]["content"].strip()
        print(f"✅ Description: {description}")
        return description
    except Exception as e:
        print(f"❌ Error generating description: {str(e)}")
        return None

def generate_image_with_sd(description):
    if sd_pipeline is None:
        print("❌ Stable Diffusion pipeline not initialized")
        return None

    prompt = (
        f"High-resolution studio photo of {description}. "
        "Ultra realistic materials, DSLR depth of field, soft shadows, subtle reflections, "
        "high-quality product photography, Canon 85mm lens, smooth background, shot on white."
    )
    negative_prompt = (
        "blurry, distorted, out of frame, watermark, overexposed, unnatural, cartoon, low resolution"
    )

    try:
        print(f"\n🎨 Generating image with prompt: {prompt}")
        print("⏳ This may take a few minutes...")
        
        # Generate the image with reduced memory usage
        image = sd_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=9,
            num_inference_steps=20,  # Reduced steps for faster generation
            width=512,
            height=512,
            num_images_per_prompt=1
        ).images[0]
        
        print("✅ Image generated successfully")
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)  # Reduced quality for smaller file size
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save the image for debugging
        image.save("generated_image.jpg")
        print("✅ Image saved as generated_image.jpg")
        
        return img_str
    except Exception as e:
        print(f"❌ Error generating image: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

# ----------- API endpoints -----------

@app.route('/api/health', methods=['GET'])
def health_check():
    if not all([waste_model, medwaste_model, ewaste_model, efficientnet_model, classification_model]):
        return jsonify({"status": "unhealthy", "message": "One or more models not loaded"}), 500
    return jsonify({"status": "healthy", "message": "All models loaded"})

@app.route('/api/process-image', methods=['POST'])
def process_image():
    if waste_model is None or medwaste_model is None or ewaste_model is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please check server logs.'
        }), 500

    try:
        # Check if image is in files or in request data
        if 'image' in request.files:
            file = request.files['image']
        elif 'file' in request.files:
            file = request.files['file']
        else:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        if not file or file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400

        image_name = file.filename
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        if '.' not in image_name or image_name.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, or WEBP)'
            }), 400

        # Read and validate image
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error reading image: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Invalid image file. Please upload a valid image.'
            }), 400
        
        # Run detections
        try:
            print("Running waste detection...")
            waste_results = waste_model(image)
            print("Running medical waste detection...")
            medwaste_results = medwaste_model(image)
            print("Running e-waste detection...")
            ewaste_results = ewaste_model(image, conf=0.25)

            waste_result = waste_results[0]
            medwaste_result = medwaste_results[0]
            ewaste_result = ewaste_results[0]
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': 'Error during object detection. Please try again.'
            }), 500
        
        # Create visualization with all detections
        try:
            print("Creating visualization...")
            # Start with the original image
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            # Define colors for different types of waste
            colors = {
                'waste': (0, 255, 0),    # Green for general waste
                'medwaste': (255, 0, 0),  # Red for medical waste
                'ewaste': (0, 0, 255)     # Blue for e-waste
            }
            
            # Process waste detections
            for box in waste_result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence >= 0.4:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = f"{WASTE_CLASS_NAMES.get(class_id, f'Unknown ({class_id})')} {confidence:.2f}"
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline=colors['waste'], width=2)
                    # Draw label background
                    text_bbox = draw.textbbox((x1, y1), label)
                    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                                 fill=colors['waste'])
                    # Draw label
                    draw.text((x1, y1), label, fill='white')

            # Process medical waste detections
            for box in medwaste_result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence >= 0.4:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = f"{MEDWASTE_CLASS_NAMES.get(class_id, f'Unknown ({class_id})')} {confidence:.2f}"
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline=colors['medwaste'], width=2)
                    # Draw label background
                    text_bbox = draw.textbbox((x1, y1), label)
                    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                                 fill=colors['medwaste'])
                    # Draw label
                    draw.text((x1, y1), label, fill='white')

            # Process e-waste detections
            for box in ewaste_result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence >= 0.25:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = f"{EWASTE_CLASS_NAMES.get(class_id, f'Unknown ({class_id})')} {confidence:.2f}"
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline=colors['ewaste'], width=2)
                    # Draw label background
                    text_bbox = draw.textbbox((x1, y1), label)
                    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                                 fill=colors['ewaste'])
                    # Draw label
                    draw.text((x1, y1), label, fill='white')
            
            # Convert to base64
            buffered = io.BytesIO()
            vis_image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': 'Error creating visualization. Please try again.'
            }), 500
        
        # Process detections for response
        detections = []
        
        # Process waste detections
        for box in waste_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence >= 0.4:
                detections.append({
                    'model': 'waste',
                    'class': class_id,
                    'class_name': WASTE_CLASS_NAMES.get(class_id, f"Unknown ({class_id})"),
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()
                })

        # Process medical waste detections
        for box in medwaste_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence >= 0.4:
                detections.append({
                    'model': 'medwaste',
                    'class': class_id,
                    'class_name': MEDWASTE_CLASS_NAMES.get(class_id, f"Unknown ({class_id})"),
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()
                })

        # Process e-waste detections
        for box in ewaste_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence >= 0.25:
                detections.append({
                    'model': 'ewaste',
                    'class': class_id,
                    'class_name': EWASTE_CLASS_NAMES.get(class_id, f"Unknown ({class_id})"),
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()
                })

        # Save results and get analysis
        try:
            print("Saving results and getting analysis...")
            filename, ai_analysis = save_detection_results(image_name, detections)
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': 'Error saving results. Please try again.'
            }), 500
        
        # Prepare response
        response = {
            'success': True,
            'image': img_str,
            'filename': filename,
            'ai_analysis': ai_analysis,
            'detections': detections,
            'summary': {
                'total_detections': len(detections),
                'waste_count': len([d for d in detections if d['model'] == 'waste']),
                'medical_waste_count': len([d for d in detections if d['model'] == 'medwaste']),
                'e_waste_count': len([d for d in detections if d['model'] == 'ewaste'])
            },
            'preview': {
                'url': f"data:image/jpeg;base64,{img_str}",
                'type': 'image/jpeg'
            }
        }
        
        print("Processing completed successfully!")
        return jsonify(response)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/classify', methods=['POST'])
def classify_image():
    if classification_model is None:
        return jsonify({
            'success': False,
            'error': 'Classification model not loaded. Please check server logs.'
        }), 500

    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Prepare image for classification
        processed_image = prepare_image_for_classification(image)
        
        # Get classification predictions
        predictions = classification_model.predict(processed_image)[0]
        
        # Debug predictions
        print("Raw predictions:", predictions)
        print("Prediction shape:", predictions.shape)
        
        # Get top 3 predictions
        top_k = 3
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_classes = [CLASSIFICATION_CLASS_NAMES[i] for i in top_indices]
        top_confidences = [float(predictions[i]) for i in top_indices]
        
        print("Top predictions:")
        for i in range(top_k):
            print(f"{top_classes[i]}: {top_confidences[i]:.4f}")
        
        # Get the highest confidence prediction
        class_id = top_indices[0]
        confidence = top_confidences[0]
        class_name = top_classes[0]
        
        # Only return if confidence is above threshold
        confidence_threshold = 0.3  # 30% confidence threshold
        if confidence < confidence_threshold:
            return jsonify({
                'success': False,
                'error': 'No confident classification found'
            }), 400
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'classification': {
                'class': class_name,
                'confidence': confidence
            },
            'image': img_str
        })

    except Exception as e:
        print(f"Error classifying image: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    if sd_pipeline is None:
        print("❌ Stable Diffusion pipeline not initialized")
        return jsonify({
            'success': False,
            'error': 'Stable Diffusion not initialized'
        }), 500

    try:
        data = request.get_json()
        if not data or 'class_name' not in data:
            print("❌ No class name provided")
            return jsonify({
                'success': False,
                'error': 'No class name provided'
            }), 400

        class_name = data['class_name']
        print(f"\n📝 Generating description for {class_name}...")
        description = generate_description_with_mistral(class_name)
        if description is None:
            print("❌ Failed to generate description")
            return jsonify({
                'success': False,
                'error': 'Failed to generate description'
            }), 500
        print(f"✅ Description: {description}")
        
        print("\n🎨 Starting image generation...")
        generated_image = generate_image_with_sd(description)
        if generated_image is None:
            print("❌ Failed to generate image")
            return jsonify({
                'success': False,
                'error': 'Failed to generate image'
            }), 500
        print("✅ Image generated successfully")
        
        return jsonify({
            'success': True,
            'generated_image': generated_image
        })

    except Exception as e:
        print(f"❌ Error in generate_image: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Internal server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("\n🚀 Starting server initialization...")
    if load_models():
        print("\n✅ Models loaded successfully")
        initialize_sd_pipeline()
        print("\n🚀 Starting Flask server...")
        app.run(debug=True)
    else:
        print("❌ Failed to load all models")
