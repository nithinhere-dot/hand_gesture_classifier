import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

st.title("🖐️ Hand Gesture Classifier")
st.write("Upload an image of a hand gesture")

model = load_model("gesture_model.h5")
class_names = np.load("data.npz", allow_pickle=True)["class_names"]

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")              # Ensure RGB
    img = img.resize((64, 64))                         # Resize to match training
    st.image(img, caption="Input", use_column_width=True)

    img_array = np.array(img) / 255.0                  # Normalize
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension

    prediction = model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    confidence = prediction[pred_index]

    st.success(f"Prediction: **{class_names[pred_index]}** ({confidence:.2f})")

    st.subheader("Confidence for all classes:")
    for i, prob in enumerate(prediction):
        st.write(f"{class_names[i]}: {prob:.2f}")
