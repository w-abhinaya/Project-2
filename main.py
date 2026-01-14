import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io


app = FastAPI(title="CIFAR-10 Image Classifier")


print("Loading model...")
model = tf.keras.models.load_model('cifar10_model_v2.keras')


class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    image = image.convert('RGB')
    image = image.resize((32, 32))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }