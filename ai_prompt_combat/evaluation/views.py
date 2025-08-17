import os
import pandas as pd
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from skimage.metrics import structural_similarity as ssim

# Set up media directory
default_media_path = os.path.join(settings.BASE_DIR, 'media')
os.makedirs(default_media_path, exist_ok=True)

# Lazy Load LLaMA 3.7 Vision Model
def get_model():
    processor = AutoProcessor.from_pretrained("meta-llama/Meta-Llama-3-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-7B-Instruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def evaluate_prompt(prompt, reference):
    """Evaluates a given prompt based on its similarity to the ideal reference prompt."""
    processor, model, device = get_model()
    inputs = processor(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    similarity_score = torch.randint(60, 100, (1,)).item()  # Mock similarity score for now
    return similarity_score

def process_csv(file_path):
    """Processes the uploaded CSV and assigns scores for each prompt."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ["Name", "Prompt_1", "Prompt_2", "Prompt_3"]
        if not all(col in df.columns for col in required_columns):
            return {"error": "Missing required columns in CSV"}
        
        reference_prompts = {
            "Prompt_1": "Write a prompt to generate an image of a peaceful beach with blue skies, palm trees, and waves.",
            "Prompt_2": "Write a prompt to generate a short poem on the importance of teachers.",
            "Prompt_3": "Write a prompt to generate a motivational message for someone preparing for exams."
        }
        
        results = []
        for _, row in df.iterrows():
            scores = [evaluate_prompt(str(row[prompt]), reference_prompts[prompt]) for prompt in reference_prompts]
            avg_score = sum(scores) / len(scores)
            results.append({"Name": row["Name"], "Score": round(avg_score, 2)})
        
        return results
    except Exception as e:
        return {"error": str(e)}

def compare_images(ideal_image_path, uploaded_image_paths):
    """Compares only the uploaded images with the ideal image using SSIM."""
    ideal_img = cv2.imread(ideal_image_path, cv2.IMREAD_GRAYSCALE)
    if ideal_img is None:
        return {"error": "Failed to load ideal image. Please check the file format and try again."}

    ideal_img = cv2.resize(ideal_img, (256, 256))
    scores = {}

    for img_path in uploaded_image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Failed to load {img_path}. Skipping...")
            continue  # Skip problematic images

        img = cv2.resize(img, (256, 256))
        similarity = ssim(ideal_img, img)
        scores[os.path.basename(img_path)] = round(similarity * 100, 2)

    return scores

@csrf_exempt
def upload_csv(request):
    """Handles CSV file uploads for Level 1 scoring."""
    if request.method == 'POST' and request.FILES.get('csv_file'):
        file = request.FILES['csv_file']
        file_path = os.path.join(default_media_path, file.name)
        with open(file_path, 'wb') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        results = process_csv(file_path)
        return render(request, 'upload.html', {"results": results, "file_uploaded": True})
    return render(request, 'upload.html', {"results": None, "file_uploaded": False})

@csrf_exempt
def upload_images(request):
    """Handles image uploads for Level 2 comparison."""
    if request.method == 'POST' and request.FILES.get('ideal_image') and request.FILES.getlist('images'):
        ideal_img = request.FILES['ideal_image']
        ideal_img_path = os.path.join(default_media_path, ideal_img.name)
        with open(ideal_img_path, 'wb') as destination:
            for chunk in ideal_img.chunks():
                destination.write(chunk)

        folder_path = os.path.join(default_media_path, "uploaded_images")
        os.makedirs(folder_path, exist_ok=True)

        uploaded_images = []
        uploaded_image_paths = []  # Keep track of newly uploaded images

        for img in request.FILES.getlist('images'):
            img_path = os.path.join(folder_path, img.name)
            with open(img_path, 'wb') as destination:
                for chunk in img.chunks():
                    destination.write(chunk)
            uploaded_images.append(img.name)
            uploaded_image_paths.append(img_path)  # Track uploaded file paths

        # Compare only the newly uploaded images
        scores = compare_images(ideal_img_path, uploaded_image_paths)
        
        return render(request, 'upload_images.html', {"scores": scores, "uploaded_images": uploaded_images, "file_uploaded": True})

    return render(request, 'upload_images.html', {"scores": None, "uploaded_images": None, "file_uploaded": False}) 