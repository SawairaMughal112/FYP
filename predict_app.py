import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, InceptionV3, ResNet50, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# Image size
IMG_SIZE = (224, 224)

# Function to load and preprocess images
def load_and_preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)  # Open the uploaded image
    img = img.resize(IMG_SIZE)  # Resize image
    img = img_to_array(img) / 255.0  # Convert image to array and normalize
    return img

# Function to build and compile a model
def build_model(base_model_name):
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    elif base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to predict anomaly on an uploaded image
def predict_image(model, img_array):
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Anomaly" if prediction >= 0.5 else "No Anomaly"

# Streamlit UI
st.title("Anomaly Detection with Different Models")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_and_preprocess_image(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    models_to_compare = ['MobileNetV2', 'InceptionV3', 'ResNet50', 'VGG16']
    predictions = []
    
    for model_name in models_to_compare:
        model = build_model(model_name)
        result = predict_image(model, img)
        predictions.append((model_name, result))
    
    st.write("Model Predictions:")
    for model_name, result in predictions:
        st.write(f"{model_name}: {result}")
