"""
FastAPI backend for Diabetic Retinopathy Classification
Provides REST API endpoints for medical image analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import numpy as np
import io
import logging
import os
from typing import List, Optional
import asyncio
from datetime import datetime

# Import our models and utilities
from models.efficientnet_model import EfficientNetDRClassifier
from models.vision_transformer import ViTDRClassifier  
from models.hybrid_model import HybridDRClassifier#, EnsembleDRClassifier
from utils.data_preprocessing import RetinalImagePreprocessor
from utils.data_augmentation import apply_test_time_augmentation
from utils.model_utils import load_model_for_inference
from config.settings import Settings
from api.schemas import PredictionResponse, ModelInfo, HealthCheck

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetic Retinopathy Classification API",
    description="Medical AI system for automated diabetic retinopathy detection and grading",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and settings
settings = Settings()
models = {}
preprocessor = None
device = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    global models, preprocessor, device

    logger.info("Starting Diabetic Retinopathy Classification API...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize preprocessor
    preprocessor = RetinalImagePreprocessor(image_size=settings.IMAGE_SIZE)

    # Load models
    try:
        if os.path.exists(settings.EFFICIENTNET_MODEL_PATH):
            models['efficientnet'] = load_model_for_inference(
                settings.EFFICIENTNET_MODEL_PATH, device
            )
            logger.info("EfficientNet model loaded successfully")

        if os.path.exists(settings.VIT_MODEL_PATH):
            models['vit'] = load_model_for_inference(
                settings.VIT_MODEL_PATH, device
            )
            logger.info("Vision Transformer model loaded successfully")

        if os.path.exists(settings.HYBRID_MODEL_PATH):
            models['hybrid'] = load_model_for_inference(
                settings.HYBRID_MODEL_PATH, device
            )
            logger.info("Hybrid model loaded successfully")

        # If no models available, create default EfficientNet
        if not models:
            logger.warning("No pre-trained models found. Using default EfficientNet.")
            models['efficientnet'] = EfficientNetDRClassifier().to(device)
            models['efficientnet'].eval()

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Fallback to default model
        models['efficientnet'] = EfficientNetDRClassifier().to(device)
        models['efficientnet'].eval()

    logger.info("API initialization completed!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Diabetic Retinopathy Classification API...")

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        models_loaded=list(models.keys()),
        device=str(device)
    )

@app.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get information about available models"""
    model_info = []

    for name, model in models.items():
        info = ModelInfo(
            name=name,
            architecture=model.__class__.__name__,
            parameters=sum(p.numel() for p in model.parameters()),
            input_size=settings.IMAGE_SIZE,
            num_classes=len(getattr(model, 'class_names', []))
        )
        model_info.append(info)

    return model_info

async def process_image(image_file: UploadFile) -> Image.Image:
    """Process uploaded image file"""

    # Validate file type
    if not image_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and process image
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Apply preprocessing
        processed_image = preprocessor.preprocess_pipeline(
            image,
            apply_ben_graham=settings.USE_BEN_GRAHAM,
            apply_clahe=settings.USE_CLAHE
        )

        return processed_image

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Error processing image file")

def create_prediction_response(predictions: dict, model_name: str, 
                             confidence_threshold: float = 0.7) -> PredictionResponse:
    """Create standardized prediction response"""

    predicted_class = int(predictions['predicted_class'][0])
    class_name = predictions['class_name'][0]
    confidence = float(predictions['confidence'][0])
    probabilities = predictions['probabilities'][0].tolist()

    # Map class to severity level
    severity_mapping = {
        0: "No Diabetic Retinopathy",
        1: "Mild Diabetic Retinopathy", 
        2: "Moderate Diabetic Retinopathy",
        3: "Severe Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy"
    }

    # Determine risk level
    if predicted_class == 0:
        risk_level = "Low"
    elif predicted_class in [1, 2]:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Clinical recommendations
    recommendations = get_clinical_recommendations(predicted_class, confidence)

    return PredictionResponse(
        predicted_class=predicted_class,
        class_name=class_name,
        severity_description=severity_mapping.get(predicted_class, "Unknown"),
        confidence=confidence,
        risk_level=risk_level,
        probabilities={
            models[list(models.keys())[0]].class_names[i] if models and hasattr(list(models.values())[0], 'class_names') else f"Class_{i}": prob 
            for i, prob in enumerate(probabilities)
        },
        model_used=model_name,
        recommendations=recommendations,
        requires_immediate_attention=predicted_class >= 3 and confidence > confidence_threshold
    )

def get_clinical_recommendations(predicted_class: int, confidence: float) -> List[str]:
    """Generate clinical recommendations based on prediction"""

    recommendations = []

    if predicted_class == 0:  # No DR
        recommendations.extend([
            "Continue regular eye examinations",
            "Maintain good blood glucose control",
            "Follow up with ophthalmologist annually"
        ])
    elif predicted_class == 1:  # Mild DR
        recommendations.extend([
            "Increase frequency of eye examinations to every 6-12 months",
            "Optimize diabetes management with endocrinologist",
            "Monitor blood pressure and cholesterol levels"
        ])
    elif predicted_class == 2:  # Moderate DR
        recommendations.extend([
            "Ophthalmologist follow-up every 3-6 months",
            "Consider fluorescein angiography",
            "Strict glycemic control is essential"
        ])
    elif predicted_class >= 3:  # Severe/Proliferative DR
        recommendations.extend([
            "URGENT: Immediate ophthalmologist consultation required",
            "Consider anti-VEGF injections or laser photocoagulation",
            "Monitor for complications (retinal detachment, glaucoma)"
        ])

    if confidence < 0.7:
        recommendations.append("Note: Prediction confidence is moderate. Consider additional clinical evaluation.")

    return recommendations

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_model(
    file: UploadFile = File(...),
    model_name: str = "efficientnet",
    use_tta: bool = False
):
    """Predict diabetic retinopathy using single model"""

    if model_name not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' not available. Available models: {list(models.keys())}"
        )

    try:
        # Process image
        image = await process_image(file)

        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        model = models[model_name]

        if use_tta:
            # Use test-time augmentation for better accuracy
            predictions = apply_test_time_augmentation(model, image_tensor.squeeze(0))
            predicted_class = torch.argmax(predictions, dim=1)
            confidence = torch.max(predictions, dim=1)[0]

            # Convert to expected format
            result = {
                'predicted_class': predicted_class.cpu().numpy(),
                'class_name': [model.class_names[predicted_class.item()]],
                'confidence': confidence.cpu().numpy(),
                'probabilities': predictions.cpu().numpy()
            }
        else:
            result = model.predict(image_tensor)

        # Create response
        response = create_prediction_response(result, model_name)

        logger.info(f"Prediction completed: {response.class_name} (confidence: {response.confidence:.3f})")

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

@app.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble(
    file: UploadFile = File(...),
    models_to_use: Optional[List[str]] = None
):
    """Predict using ensemble of multiple models"""

    if models_to_use is None:
        models_to_use = list(models.keys())

    # Validate requested models
    invalid_models = [m for m in models_to_use if m not in models]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid models: {invalid_models}. Available: {list(models.keys())}"
        )

    try:
        # Process image
        image = await process_image(file)

        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Collect predictions from all models
        all_predictions = []

        for model_name in models_to_use:
            model = models[model_name]
            with torch.no_grad():
                logits = model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                all_predictions.append(probabilities)

        # Average ensemble
        ensemble_probs = torch.mean(torch.stack(all_predictions), dim=0)
        predicted_class = torch.argmax(ensemble_probs, dim=1)
        confidence = torch.max(ensemble_probs, dim=1)[0]

        # Format result
        result = {
            'predicted_class': predicted_class.cpu().numpy(),
            'class_name': [models[models_to_use[0]].class_names[predicted_class.item()]],
            'confidence': confidence.cpu().numpy(),
            'probabilities': ensemble_probs.cpu().numpy()
        }

        response = create_prediction_response(result, f"ensemble({','.join(models_to_use)})")

        logger.info(f"Ensemble prediction completed: {response.class_name} (confidence: {response.confidence:.3f})")

        return response

    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during ensemble prediction")

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), model_name: str = "efficientnet"):
    """Batch prediction for multiple images"""

    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size too large. Maximum allowed: {settings.MAX_BATCH_SIZE}"
        )

    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available"
        )

    results = []

    for i, file in enumerate(files):
        try:
            # Process each image
            response = await predict_single_model(file, model_name)
            response.filename = file.filename
            results.append(response)

        except Exception as e:
            logger.error(f"Error processing file {i}: {e}")
            # Continue with other files, but log the error
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })

    return {"batch_results": results, "total_processed": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
