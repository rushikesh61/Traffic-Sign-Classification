import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrafficSign AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "traffic_sign_cnn_model.keras"

CLASS_NAMES = {
    0: "Speed Limit 20 km/h",
    1: "Speed Limit 30 km/h",
    2: "Speed Limit 50 km/h",
    3: "Speed Limit 60 km/h",
    4: "Speed Limit 70 km/h",
    5: "Speed Limit 80 km/h",
    6: "End of Speed Limit 80 km/h",
    7: "Speed Limit 100 km/h",
    8: "Speed Limit 120 km/h",
    9: "No Passing",
    10: "No Passing for Vehicles > 3.5 Tons",
    11: "Right-of-Way at Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicles > 3.5 Tons Prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve Left",
    20: "Dangerous Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on Right",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Speed & Passing Limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout Mandatory",
    41: "End of No Passing",
    42: "End of No Passing for Vehicles > 3.5 Tons"
}

# ============================================================
# MODEL LOAD
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)


# ============================================================
# FUNCTIONS
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((32, 32))

    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict_image(image):

    processed = preprocess_image(image)

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[::-1][:3]

    results = []

    for index in top_indices:

        results.append({
            "id": int(index),
            "name": CLASS_NAMES[int(index)],
            "confidence": float(prediction[index])
        })

    return results


def sign_icon(class_id):

    if class_id in range(0, 9):
        return "🔴"

    if class_id in [13, 14]:
        return "🛑"

    if class_id == 17:
        return "🚫"

    if class_id in [
        18, 19, 20, 21, 22,
        23, 25, 27, 28, 29,
        30, 31
    ]:
        return "⚠️"

    if class_id in [
        33, 34, 35, 36,
        37, 38, 39, 40
    ]:
        return "🔵"

    return "🚦"


def driving_guidance(class_id):

    guidance = {

        0: "Reduce speed to 20 km/h.",
        1: "Maintain a maximum speed of 30 km/h.",
        2: "Maintain a maximum speed of 50 km/h.",
        3: "Maintain a maximum speed of 60 km/h.",
        4: "Maintain a maximum speed of 70 km/h.",
        5: "Maintain a maximum speed of 80 km/h.",
        6: "The 80 km/h restriction has ended.",
        7: "Maintain a maximum speed of 100 km/h.",
        8: "Maintain a maximum speed of 120 km/h.",
        9: "Do not overtake other vehicles.",
        10: "Heavy vehicles are not allowed to overtake.",
        11: "Follow the right-of-way rules.",
        12: "You are travelling on a priority road.",
        13: "Slow down and give way.",
        14: "Stop completely before proceeding.",
        15: "Vehicles are prohibited in this area.",
        16: "Vehicles over 3.5 tons are prohibited.",
        17: "Entry is prohibited from this direction.",
        18: "Drive carefully and watch for hazards.",
        19: "Prepare for a dangerous curve to the left.",
        20: "Prepare for a dangerous curve to the right.",
        21: "Reduce speed before consecutive curves.",
        22: "Drive carefully on the uneven road.",
        23: "Reduce speed because the road may be slippery.",
        24: "The road narrows on the right.",
        25: "Road construction or maintenance is ahead.",
        26: "Traffic signals are ahead.",
        27: "Watch carefully for pedestrians.",
        28: "Watch carefully for children crossing.",
        29: "Watch carefully for bicycles.",
        30: "Be alert for ice or snow.",
        31: "Watch for wild animals crossing.",
        32: "Previous speed and passing restrictions have ended.",
        33: "Turn right ahead.",
        34: "Turn left ahead.",
        35: "Continue straight ahead.",
        36: "You may go straight or turn right.",
        37: "You may go straight or turn left.",
        38: "Keep to the right.",
        39: "Keep to the left.",
        40: "Follow the roundabout direction.",
        41: "The no-passing restriction has ended.",
        42: "The heavy-vehicle no-passing restriction has ended."
    }

    return guidance.get(
        class_id,
        "Follow the road signs and drive carefully."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚦 TrafficSign AI")

    st.caption("CNN CLASSIFICATION SYSTEM")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select Page",
        [
            "🏠 Dashboard",
            "🔍 Predict Sign",
            "📊 Model Performance",
            "🚦 Sign Classes",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    if model_loaded:
        st.success("● MODEL ONLINE")
    else:
        st.error("● MODEL ERROR")

    st.divider()

    st.caption("SYSTEM INFO")

    st.write("⚙️ TensorFlow")
    st.write("🧠 Keras CNN")
    st.write("🖼️ Input: 32 × 32 RGB")
    st.write("🚦 Classes: 43")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🚦 Traffic Sign Classification AI")

    st.subheader(
        "Deep Learning • Computer Vision • CNN"
    )

    st.info(
        "An intelligent image classification system that "
        "recognizes traffic signs using a Convolutional Neural Network."
    )

    st.divider()

    st.header("📊 Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Traffic Classes",
            "43",
            "Categories"
        )

    with c2:
        st.metric(
            "Training Images",
            "39,209",
            "Images"
        )

    with c3:
        st.metric(
            "Test Images",
            "12,630",
            "Images"
        )

    with c4:
        st.metric(
            "Test Accuracy",
            "96.44%",
            "CNN"
        )

    st.divider()

    st.header("🧠 CNN Architecture")

    a1, a2, a3, a4, a5 = st.columns(5)

    with a1:
        st.info("🖼️ INPUT\n\n32 × 32 × 3")

    with a2:
        st.info("🧩 CONVOLUTION\n\n32 Filters")

    with a3:
        st.info("🧠 CONVOLUTION\n\n64 + 128 Filters")

    with a4:
        st.info("🔄 DENSE\n\n128 Neurons")

    with a5:
        st.success("🚦 OUTPUT\n\n43 Classes")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("🎯 Project Objective")

        st.write(
            "The objective of this project is to automatically "
            "recognize and classify road traffic signs from images "
            "using a Convolutional Neural Network."
        )

        st.write(
            "The CNN learns visual patterns such as shape, color, "
            "symbols and structures from traffic-sign images."
        )

    with right:

        st.subheader("⚡ System Status")

        if model_loaded:

            st.success(
                "Model is loaded and ready for prediction."
            )

        else:

            st.error(
                "Model could not be loaded."
            )

    st.divider()

    st.subheader("🚀 Quick Start")

    st.write(
        "Go to **Predict Sign** from the sidebar and upload "
        "a traffic-sign image."
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔍 Predict Sign":

    st.title("🔍 Traffic Sign Prediction")

    st.write(
        "Upload an image and let the trained CNN identify "
        "the traffic-sign category."
    )

    st.divider()

    if not model_loaded:

        st.error(
            "Model could not be loaded."
        )

        st.code(model_error)

    else:

        left, right = st.columns(
            [1, 1],
            gap="large"
        )

        with left:

            st.subheader("📤 Upload Image")

            uploaded_file = st.file_uploader(
                "Choose a traffic sign image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

            st.divider()

            threshold = st.slider(
                "Confidence Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.70,
                step=0.01
            )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            with left:

                st.image(
                    image,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            results = predict_image(image)

            best = results[0]

            with right:

                st.subheader("🤖 AI Prediction")

                st.metric(
                    "Predicted Sign",
                    best["name"]
                )

                st.metric(
                    "Confidence",
                    f"{best['confidence'] * 100:.2f}%"
                )

                if best["confidence"] >= threshold:

                    st.success(
                        "✓ High-confidence prediction"
                    )

                else:

                    st.warning(
                        "⚠ Low-confidence prediction"
                    )

                st.divider()

                st.subheader("💡 Driving Guidance")

                st.info(
                    driving_guidance(
                        best["id"]
                    )
                )

            st.divider()

            st.header("📊 Top 3 Predictions")

            for rank, result in enumerate(
                results,
                start=1
            ):

                col1, col2, col3 = st.columns(
                    [0.10, 0.65, 0.25]
                )

                with col1:

                    st.subheader(
                        f"#{rank}"
                    )

                with col2:

                    st.write(
                        f"{sign_icon(result['id'])} "
                        f"**{result['name']}**"
                    )

                    st.progress(
                        result["confidence"]
                    )

                with col3:

                    st.write(
                        f"**{result['confidence'] * 100:.2f}%**"
                    )

        else:

            st.info(
                "📷 Upload an image to start prediction."
            )

            st.write(
                "Supported formats: JPG, JPEG, PNG"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Performance of the CNN model on the independent test dataset."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Test Accuracy",
            "96.44%"
        )

    with c2:

        st.metric(
            "Test Error",
            "3.56%"
        )

    with c3:

        st.metric(
            "Macro F1",
            "94.41%"
        )

    with c4:

        st.metric(
            "Weighted F1",
            "96.41%"
        )

    st.divider()

    st.header("🧠 Model Configuration")

    config1, config2 = st.columns(2)

    with config1:

        st.write("**Architecture:** CNN")
        st.write("**Input Shape:** 32 × 32 × 3")
        st.write("**Output Classes:** 43")
        st.write("**Optimizer:** Adam")

    with config2:

        st.write(
            "**Loss:** Sparse Categorical Crossentropy"
        )

        st.write("**Epochs:** 15")

        st.write("**Batch Size:** 64")

        st.write(
            "**Activation:** ReLU + Softmax"
        )

    st.divider()

    st.header("📈 Test Dataset Results")

    st.write(
        "Total Test Images: **12,630**"
    )

    st.write(
        "Correct Predictions: **12,180**"
    )

    st.write(
        "Incorrect Predictions: **450**"
    )

    st.progress(
        0.9644,
        text="Test Accuracy — 96.44%"
    )

    st.divider()

    st.header("🔬 CNN Processing Pipeline")

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.info("🖼️ Image\n\nInput")

    with p2:
        st.info("🧩 Conv2D\n\nFeature Extraction")

    with p3:
        st.info("🔄 Pooling\n\nDownsampling")

    with p4:
        st.info("🧠 Dense\n\nClassification")

    with p5:
        st.success("🚦 Softmax\n\nPrediction")


# ============================================================
# SIGN CLASSES
# ============================================================

elif page == "🚦 Sign Classes":

    st.title("🚦 Traffic Sign Classes")

    st.write(
        "The CNN model recognizes 43 traffic-sign categories."
    )

    search = st.text_input(
        "🔎 Search class",
        placeholder="Try: speed, road, stop, curve..."
    )

    st.divider()

    filtered_classes = []

    for class_id, name in CLASS_NAMES.items():

        if search.lower() in name.lower():

            filtered_classes.append(
                (class_id, name)
            )

    if not filtered_classes:

        st.warning(
            "No matching class found."
        )

    else:

        for start in range(
            0,
            len(filtered_classes),
            3
        ):

            row = filtered_classes[
                start:start + 3
            ]

            cols = st.columns(3)

            for col, item in zip(
                cols,
                row
            ):

                class_id, name = item

                with col:

                    st.info(
                        f"{sign_icon(class_id)} "
                        f"**Class {class_id}**\n\n"
                        f"{name}"
                    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About TrafficSign AI")

    st.subheader(
        "Final Year Deep Learning Project"
    )

    st.divider()

    st.header("🎯 Problem Statement")

    st.write(
        "Traffic signs provide important information to drivers "
        "and intelligent transportation systems. Automatic "
        "recognition of these signs can help computer vision "
        "systems understand road environments."
    )

    st.divider()

    st.header("💡 Proposed Solution")

    st.write(
        "This project uses a Convolutional Neural Network to "
        "classify traffic-sign images into 43 different categories."
    )

    st.write(
        "The input image is resized to 32 × 32 pixels and normalized "
        "before being passed through the trained CNN."
    )

    st.divider()

    st.header("🛠️ Technologies")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.success("🐍 Python")

    with tech2:
        st.success("🧠 TensorFlow")

    with tech3:
        st.success("🔬 Keras")

    with tech4:
        st.success("🌐 Streamlit")

    st.divider()

    st.header("🚀 Project Workflow")

    st.write("1️⃣ Dataset Collection")
    st.write("2️⃣ Image Preprocessing")
    st.write("3️⃣ Train / Validation Split")
    st.write("4️⃣ CNN Model Training")
    st.write("5️⃣ Model Evaluation")
    st.write("6️⃣ Streamlit Deployment")

    st.divider()

    st.header("📌 Project Summary")

    st.success(
        "CNN-based Traffic Sign Classification system "
        "with 43 classes and 96.44% test accuracy."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚦 TrafficSign AI  •  Deep Learning  •  CNN  •  "
    "Computer Vision  •  43 Classes  •  Academic Project"
)
