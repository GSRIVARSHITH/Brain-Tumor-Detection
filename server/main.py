from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from server.api.predict import BrainTumorPredictor




app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ONCE ----------------
CLASSES = ["Glioma", "Meningioma", "No_Tumor", "Pituitary"]

predictor = BrainTumorPredictor(
    model_path="model/cnn_model.h5",
    classes=CLASSES
)

import tempfile

# ---------------- API ----------------
@app.post("/predict")
def predict_image(file: UploadFile = File(...)):
    # Create a temporary file to save the uploaded image
    # delete=False because we need to close it before passing path to cv2 (on Windows)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        image = predictor.load_and_preprocess_image(tmp_path)
        result = predictor.predict(image)
        return result
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
