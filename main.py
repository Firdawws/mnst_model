import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained MNIST model
model = load_model("mnist_model.keras")

app = FastAPI()

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # Shape (1, 28, 28)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_array = preprocess_image(contents)
    predictions = model.predict(image_array)
    predicted_digit = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    return JSONResponse(content={"digit": predicted_digit, "confidence": confidence})

@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST API. Upload a digit to /predict for classification."}
