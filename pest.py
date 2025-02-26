#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from deep_translator import GoogleTranslator

# ✅ Path to your model (Make sure this is correct)
model_path = "/Users/amartyajethmalani/Desktop/pest_classification_model.keras"

if not os.path.exists(model_path):
    st.error(f"🚨 Model file not found: {model_path}")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ✅ Define class labels
class_labels = [
    "Brown Planthopper",
    "Cotton Bollworm",
    "Oriental Fruit Fly",
    "Shoot and Fruit Borer",
    "Swarming Caterpillar",
    "Whitefly"
]

# ✅ Pest-specific treatments
treatments = {
    "Brown Planthopper": "Use resistant rice varieties and apply systemic insecticides.",
    "Cotton Bollworm": "Use pheromone traps and apply insecticides like Spinosad.",
    "Oriental Fruit Fly": "Use protein-bait sprays and bagging for fruit protection.",
    "Shoot and Fruit Borer": "Use crop rotation and apply biopesticides like Beauveria bassiana.",
    "Swarming Caterpillar": "Apply Bacillus thuringiensis (Bt) or neem-based bio-pesticides.",
    "Whitefly": "Apply neem oil or introduce predatory insects."
}

# ✅ Streamlit Web App
st.set_page_config(page_title="Pest Identification", layout="centered")
st.title("🌱 Pest Detection for Farmers 🌾")

uploaded_file = st.file_uploader("📸 Upload an image of the pest:", type=["jpg", "png", "jpeg"])

language = st.selectbox("🌍 Select Language:", ["English", "Hindi", "Marathi"])
language_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}

if uploaded_file:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    
    # ✅ Save file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Process image
    img = load_img(temp_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    pest_name = class_labels[predicted_class]

    # ✅ Translate pest name and treatment
    pest_name_translated = GoogleTranslator(source="auto", target=language_map[language]).translate(pest_name)
    treatment_translated = GoogleTranslator(source="auto", target=language_map[language]).translate(
        treatments.get(pest_name, "No treatment information available.")
    )

    # ✅ Display results
    st.success(f"🐛 **Pest Identified:** {pest_name_translated}")
    st.info(f"💡 **Recommended Treatment:** {treatment_translated}")


# In[4]:


jupyter nbconvert --to script pest.ipynb


# In[ ]:




