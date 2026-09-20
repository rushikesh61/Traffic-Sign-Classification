
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

# -------------------------------------------------
# Traffic Sign Class Names
# -------------------------------------------------

class_names = {
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------

MODEL_PATH = "traffic_sign_cnn_model.keras"

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_trained_model()

# -------------------------------------------------
# Header
# -------------------------------------------------

st.title("🚦 Traffic Sign Classification")
st.write(
    "Upload a traffic sign image and the CNN model "
    "will predict the traffic sign."
)

st.divider()

# -------------------------------------------------
# File Upload
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Traffic Sign",
        width=300
    )

    # Resize image
    image_resized = image.resize((32, 32))

    # Convert to NumPy array
    image_array = np.array(image_resized)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    predicted_name = class_names[predicted_class]

    st.success(f"Prediction: {predicted_name}")

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    st.info(f"Class ID: {predicted_class}")

st.divider()

st.caption(
    "Traffic Sign Classification using Convolutional Neural Network (CNN)"
)
