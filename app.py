from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

app = Flask(__name__)
CORS(app)

# ✅ Load model
model = tf.keras.models.load_model("agrolens_model.h5")

# ✅ Correct class labels (15 classes)
class_labels = [
    'Pepper__bell___healthy',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Target_Spot',
    'Pepper__bell___Bacterial_spot',
    'Tomato_Leaf_Mold',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Bacterial_spot',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Late_blight',
    'Potato___Late_blight',
    'Potato___Early_blight',
    'Tomato_Early_blight',
    'Tomato__Tomato_mosaic_virus',
    'Potato___healthy',
    'Tomato_healthy'
]

# ✅ Remedies dictionary
remedies = {
    "Tomato__Early_blight": "Apply copper fungicide. Avoid wetting leaves.",
    "Potato___Late_blight": "Spray mancozeb. Keep foliage dry.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Remove infected plants. Use yellow sticky traps.",
    "Tomato_Bacterial_spot": "Apply copper sprays weekly. Avoid moisture.",
    "Tomato_healthy": "No disease detected. Maintain regular care.",
    "Potato___healthy": "Healthy plant. No action required.",
    "default": "Use organic pesticide. Monitor weekly."
}

# ✅ Weather by coordinates
def get_weather_by_coords(lat, lon):
    API_KEY = "8800fe004d2596e3f2164493b1656591"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()
        temp = res['main']['temp']
        humidity = res['main']['humidity']
        description = res['weather'][0]['description']
        return f"Temperature: {temp}°C, Humidity: {humidity}%, Weather: {description}"
    except:
        return "Weather unavailable"

# ✅ Translator
def translate(text, lang):
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except:
        return text  # fallback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    lang = request.form.get('lang', 'en')
    lat = request.form.get('lat', None)
    lon = request.form.get('lon', None)

    # ✅ Preprocess image
    img = Image.open(file).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # ✅ Model prediction
    preds = model.predict(img_array)
    print("📊 Prediction vector length:", len(preds[0]))
    print("🔠 Class labels length:", len(class_labels))

    # ✅ Debug all predictions
    for i, prob in enumerate(preds[0]):
        if i < len(class_labels):
            print(f"{class_labels[i]}: {prob:.4f}")

    # ✅ Final prediction
    pred_index = int(np.argmax(preds))
    pred_class = class_labels[pred_index]
    print("🔍 Predicted class:", pred_class)

    # ✅ Remedy
    remedy = remedies.get(pred_class, remedies["default"])

    # ✅ Weather
    weather = get_weather_by_coords(lat, lon)

    # ✅ Translate
    translated_disease = translate(pred_class.replace("_", " "), lang)
    translated_remedy = translate(remedy, lang)
    translated_weather = translate(weather, lang)

    # ✅ Voice
    voice_text = f"{translated_disease}. {translated_remedy}. {translated_weather}"
    tts = gTTS(voice_text, lang=lang)
    tts.save("static/voice.mp3")

    return render_template('index.html',
                           prediction=translated_disease,
                           remedy=translated_remedy,
                           weather=translated_weather,
                           lang=lang,
                           voice_url=url_for('static', filename='voice.mp3'))

if __name__ == '__main__':
    app.run(debug=True)
