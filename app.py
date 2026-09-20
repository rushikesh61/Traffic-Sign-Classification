import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="TrafficSign AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS USING st.html()
# =========================================================
st.html("""
<style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1f35 45%, #06101c 100%);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081525 0%, #0d2238 100%);
        border-right: 1px solid #1e4f73;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .hero {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0d2d4a, #0a1930);
        border: 1px solid #1d638c;
        box-shadow: 0 10px 35px rgba(0, 180, 255, 0.12);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin: 0;
        color: #38bdf8;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 17px;
        margin-top: 8px;
    }

    .card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(15, 35, 55, 0.88);
        border: 1px solid #245b7a;
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 21px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 8px;
    }

    .prediction {
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(135deg, #092f46, #0b2034);
        border: 2px solid #22d3ee;
        box-shadow: 0 0 25px rgba(34, 211, 238, 0.12);
        text-align: center;
    }

    .prediction h2 {
        color: #67e8f9;
        margin-bottom: 5px;
    }

    .confidence {
        font-size: 34px;
        font-weight: 800;
        color: #38bdf8;
    }

    .instruction {
        padding: 18px;
        border-radius: 15px;
        background: #102d42;
        border-left: 5px solid #22d3ee;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px 0 10px;
    }

    .camera-box {
        padding: 18px;
        border-radius: 15px;
        background: #0c2438;
        border: 1px solid #245b7a;
    }

    .small-text {
        color: #94a3b8;
        font-size: 14px;
    }
</style>
""")

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_cnn_model.keras")

model = load_model()

# =========================================================
# CLASS NAMES
# =========================================================
class_names = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]

# =========================================================
# DRIVING GUIDANCE
# =========================================================
guidance = {
    "Stop": "Come to a complete stop before proceeding.",
    "Yield": "Slow down and give priority to other road users.",
    "No entry": "Do not enter this road or area.",
    "Speed limit (20 km/h)": "Keep your speed at or below 20 km/h.",
    "Speed limit (30 km/h)": "Keep your speed at or below 30 km/h.",
    "Speed limit (50 km/h)": "Keep your speed at or below 50 km/h.",
    "Speed limit (60 km/h)": "Keep your speed at or below 60 km/h.",
    "Speed limit (70 km/h)": "Keep your speed at or below 70 km/h.",
    "Speed limit (80 km/h)": "Keep your speed at or below 80 km/h.",
    "Speed limit (100 km/h)": "Keep your speed at or below 100 km/h.",
    "Speed limit (120 km/h)": "Keep your speed at or below 120 km/h.",
    "No passing": "Overtaking is not allowed in this area.",
    "Keep right": "Keep to the right side of the road.",
    "Keep left": "Keep to the left side of the road.",
    "Turn right ahead": "Prepare to turn right ahead.",
    "Turn left ahead": "Prepare to turn left ahead.",
    "Ahead only": "Continue straight ahead.",
    "Roundabout mandatory": "Follow the roundabout direction.",
    "Traffic signals": "Watch for traffic lights and follow their signals.",
    "Road work": "Slow down and watch for road construction.",
    "Pedestrians": "Watch carefully for pedestrians.",
    "Children crossing": "Slow down and watch for children crossing.",
    "Dangerous curve left": "Reduce speed and prepare for a left curve.",
    "Dangerous curve right": "Reduce speed and prepare for a right curve.",
    "Slippery road": "Drive carefully because the road may be slippery.",
    "Beware of ice/snow": "Reduce speed and drive carefully in icy or snowy conditions.",
    "Wild animals crossing": "Watch for animals crossing the road.",
    "General caution": "Drive carefully and be alert to road conditions."
}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🚦 TrafficSign AI")
    st.caption("CNN Based Traffic Sign Classification")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Predict Sign",
            "📊 Model Performance",
            "🚦 Sign Classes",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.info(
        "AI Model: Custom CNN\n\n"
        "Classes: 43\n"
        "Test Accuracy: 96.44%"
    )

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.html("""
    <div class="hero">
        <h1>🚦 TrafficSign AI</h1>
        <p>Intelligent Traffic Sign Classification using Convolutional Neural Networks</p>
    </div>
    """)

    st.subheader("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Traffic Classes", "43")

    with c2:
        st.metric("Training Images", "39,209")

    with c3:
        st.metric("Test Images", "12,630")

    with c4:
        st.metric("Test Accuracy", "96.44%")

    st.divider()

    st.subheader("🧠 CNN Architecture")

    cols = st.columns(5)

    architecture = [
        ("1", "Conv2D", "32 Filters"),
        ("2", "Conv2D", "64 Filters"),
        ("3", "Conv2D", "128 Filters"),
        ("4", "Dense", "128 Neurons"),
        ("5", "Output", "43 Classes")
    ]

    for col, item in zip(cols, architecture):
        with col:
            st.info(f"**Layer {item[0]}**\n\n{item[1]}\n\n{item[2]}")

    st.divider()

    st.subheader("✨ Main Features")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.success("📷 Camera Prediction\n\nCapture a traffic sign directly using your camera.")

    with f2:
        st.success("📁 Image Upload\n\nUpload JPG, JPEG or PNG traffic sign images.")

    with f3:
        st.success("📊 Smart Analysis\n\nGet prediction confidence and top 3 results.")

# =========================================================
# PREDICT SIGN
# =========================================================
elif page == "🔍 Predict Sign":

    st.html("""
    <div class="hero">
        <h1>🔍 Predict Traffic Sign</h1>
        <p>Capture an image or upload a traffic sign and let the CNN model classify it.</p>
    </div>
    """)

    st.subheader("📥 Choose Input Method")

    input_method = st.radio(
        "Select how you want to provide the image:",
        ["📷 Open Camera", "📁 Upload File"],
        horizontal=True
    )

    image = None

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------
    if input_method == "📷 Open Camera":

        st.html("""
        <div class="camera-box">
            <b>📷 Camera Mode</b><br>
            <span class="small-text">
            Allow camera permission and capture a clear traffic sign.
            </span>
        </div>
        """)

        camera_image = st.camera_input(
            "Take a picture of the traffic sign"
        )

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------
    else:

        uploaded_file = st.file_uploader(
            "Upload Traffic Sign Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image containing a traffic sign."
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------
    if image is not None:

        st.divider()

        left, right = st.columns([1, 1.25])

        with left:

            st.subheader("🖼️ Image Preview")

            st.image(
                image,
                caption="Input Traffic Sign",
                use_container_width=True
            )

        with right:

            st.subheader("⚙️ Prediction Settings")

            threshold = st.slider(
                "Confidence Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.70,
                step=0.01
            )

            st.caption(
                f"Prediction will be marked reliable when confidence is ≥ {threshold:.0%}"
            )

            if st.button(
                "🚀 Predict Traffic Sign",
                type="primary",
                use_container_width=True
            ):

                # Resize and normalize
                img = image.resize((32, 32))

                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                predictions = model.predict(
                    img_array,
                    verbose=0
                )[0]

                # Top prediction
                top_indices = np.argsort(predictions)[::-1][:3]

                predicted_index = top_indices[0]
                predicted_name = class_names[predicted_index]
                confidence = float(predictions[predicted_index])

                st.session_state["prediction_done"] = True
                st.session_state["predicted_name"] = predicted_name
                st.session_state["confidence"] = confidence
                st.session_state["top_indices"] = top_indices
                st.session_state["predictions"] = predictions

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------
        if st.session_state.get("prediction_done", False):

            predicted_name = st.session_state["predicted_name"]
            confidence = st.session_state["confidence"]
            top_indices = st.session_state["top_indices"]
            predictions = st.session_state["predictions"]

            st.divider()

            st.subheader("🎯 Prediction Result")

            r1, r2 = st.columns([1.5, 1])

            with r1:

                st.html(f"""
                <div class="prediction">
                    <h2>🚦 {predicted_name}</h2>
                    <div class="confidence">{confidence:.2%}</div>
                    <p>Model Confidence</p>
                </div>
                """)

            with r2:

                if confidence >= threshold:
                    st.success(
                        f"✅ High Confidence\n\n"
                        f"The model is confident about this prediction."
                    )
                else:
                    st.warning(
                        f"⚠️ Low Confidence\n\n"
                        f"Try a clearer or closer image."
                    )

            # -------------------------------------------------
            # DRIVING GUIDANCE
            # -------------------------------------------------
            st.subheader("🚘 Driving Guidance")

            instruction = guidance.get(
                predicted_name,
                "Follow the traffic rules associated with this sign."
            )

            st.html(f"""
            <div class="instruction">
                <b>💡 Recommended Action</b><br><br>
                {instruction}
            </div>
            """)

            # -------------------------------------------------
            # TOP 3
            # -------------------------------------------------
            st.divider()

            st.subheader("🏆 Top 3 Predictions")

            for rank, idx in enumerate(top_indices, start=1):

                name = class_names[idx]
                score = float(predictions[idx])

                c1, c2 = st.columns([2, 4])

                with c1:
                    st.write(f"**#{rank}  {name}**")

                with c2:
                    st.progress(
                        min(score, 1.0),
                        text=f"{score:.2%}"
                    )

            # -------------------------------------------------
            # RESET
            # -------------------------------------------------
            st.divider()

            if st.button(
                "🔄 Clear Prediction",
                use_container_width=True
            ):
                st.session_state["prediction_done"] = False
                st.rerun()

    else:

        st.info(
            "👆 Select **Open Camera** or **Upload File** above to start prediction."
        )

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "📊 Model Performance":

    st.html("""
    <div class="hero">
        <h1>📊 Model Performance</h1>
        <p>Evaluation results of the trained CNN model.</p>
    </div>
    """)

    st.subheader("📈 Test Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Accuracy", "96.44%")

    with c2:
        st.metric("Error Rate", "3.56%")

    with c3:
        st.metric("Macro F1", "94.41%")

    with c4:
        st.metric("Weighted F1", "96.41%")

    st.divider()

    st.subheader("⚙️ Model Configuration")

    st.info(
        """
        **Input Size:** 32 × 32 × 3  
        **Architecture:** Convolutional Neural Network  
        **Optimizer:** Adam  
        **Loss:** Sparse Categorical Crossentropy  
        **Epochs:** 15  
        **Batch Size:** 64  
        **Classes:** 43
        """
    )

    st.subheader("🔄 Processing Pipeline")

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.info("📷 Image")

    with p2:
        st.info("📐 Resize")

    with p3:
        st.info("⚖️ Normalize")

    with p4:
        st.info("🧠 CNN")

    with p5:
        st.info("🎯 Prediction")

# =========================================================
# SIGN CLASSES
# =========================================================
elif page == "🚦 Sign Classes":

    st.html("""
    <div class="hero">
        <h1>🚦 Traffic Sign Classes</h1>
        <p>Explore all 43 traffic sign categories recognized by the CNN model.</p>
    </div>
    """)

    search = st.text_input(
        "🔎 Search Traffic Sign",
        placeholder="Example: speed, stop, curve, road..."
    )

    filtered_classes = [
        (i, name)
        for i, name in enumerate(class_names)
        if search.lower() in name.lower()
    ]

    st.write(
        f"Showing **{len(filtered_classes)}** of **{len(class_names)}** classes"
    )

    for idx, name in filtered_classes:

        col1, col2 = st.columns([1, 5])

        with col1:
            st.info(f"Class {idx}")

        with col2:
            st.write(f"🚦 **{name}**")

# =========================================================
# ABOUT
# =========================================================
elif page == "ℹ️ About Project":

    st.html("""
    <div class="hero">
        <h1>ℹ️ About TrafficSign AI</h1>
        <p>A Deep Learning based traffic sign classification system.</p>
    </div>
    """)

    st.subheader("🎓 Project Objective")

    st.write(
        """
        The objective of this project is to develop a Deep Learning model
        that can automatically recognize and classify traffic signs from
        images.
        """
    )

    st.subheader("🧠 Technology Stack")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.info("Python")

    with t2:
        st.info("TensorFlow")

    with t3:
        st.info("CNN")

    with t4:
        st.info("Streamlit")

    st.subheader("📊 Dataset")

    st.write(
        """
        The model is trained on a traffic sign dataset containing
        **43 different traffic sign classes**.
        """
    )

    st.subheader("🚀 Deployment")

    st.success(
        "This application provides real-time image-based traffic sign prediction "
        "through camera capture and image upload."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "🚦 TrafficSign AI | CNN Based Traffic Sign Classification | "
    "Deep Learning Final Year Project"
)
