import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TrafficSign AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.html("""
<style>

    /* ========================================================
       GLOBAL APP
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(14, 165, 233, 0.12),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #06111f 0%,
                #0a1929 50%,
                #07111e 100%
            );

        color: #f8fafc !important;
    }


    /* Main text */

    .stApp p,
    .stApp span,
    .stApp label {
        color: #e2e8f0;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #071525 0%,
                #0b2035 100%
            ) !important;

        border-right: 1px solid #1e5274;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
    }


    /* Sidebar radio */

    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #f8fafc !important;
        background: transparent !important;
    }


    /* ========================================================
       HEADINGS
    ======================================================== */

    h1 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #67e8f9 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #bae6fd !important;
        font-weight: 700 !important;
    }


    /* ========================================================
       HERO SECTION
    ======================================================== */

    .hero {
        padding: 32px;
        margin-bottom: 25px;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                #0c304b,
                #0a1c31
            );

        border: 1px solid #216489;

        box-shadow:
            0 12px 40px rgba(0, 180, 255, 0.12);
    }

    .hero-title {
        color: #38bdf8 !important;
        font-size: 42px;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #dbeafe !important;
        font-size: 17px;
    }


    /* ========================================================
       METRIC CARDS
    ======================================================== */

    [data-testid="stMetric"] {
        background: rgba(15, 42, 64, 0.95) !important;

        border: 1px solid #285e7d !important;

        border-radius: 16px !important;

        padding: 18px !important;

        box-shadow:
            0 5px 20px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {
        width: 100%;

        min-height: 45px;

        border-radius: 11px !important;

        border: 1px solid #38bdf8 !important;

        background:
            linear-gradient(
                90deg,
                #0284c7,
                #0891b2
            ) !important;

        color: #ffffff !important;

        font-weight: 750 !important;

        box-shadow:
            0 5px 15px rgba(14, 165, 233, 0.18);
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                90deg,
                #0369a1,
                #0e7490
            ) !important;

        border-color: #67e8f9 !important;

        color: #ffffff !important;
    }


    /* ========================================================
       RADIO
    ======================================================== */

    [data-testid="stRadio"] label {
        color: #f8fafc !important;
    }

    [data-testid="stRadio"] p {
        color: #f8fafc !important;
    }


    /* ========================================================
       FILE UPLOADER
    ======================================================== */

    [data-testid="stFileUploader"] {
        background: #102a40 !important;

        border: 1px solid #286789 !important;

        border-radius: 16px !important;

        padding: 12px !important;
    }

    [data-testid="stFileUploader"] * {
        color: #f8fafc !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0b2439 !important;

        border: 1px dashed #3b82f6 !important;
    }


    /* ========================================================
       CAMERA
    ======================================================== */

    [data-testid="stCameraInput"] {
        background: #102a40 !important;

        border: 1px solid #286789 !important;

        border-radius: 16px !important;

        padding: 12px !important;
    }

    [data-testid="stCameraInput"] * {
        color: #ffffff !important;
    }


    /* ========================================================
       INPUT BOXES
    ======================================================== */

    input,
    textarea {
        background: #102a40 !important;

        color: #ffffff !important;

        border-color: #286789 !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #94a3b8 !important;
    }


    /* ========================================================
       SEARCH BOX
    ======================================================== */

    [data-testid="stTextInput"] input {
        background: #102a40 !important;

        color: #ffffff !important;

        border: 1px solid #286789 !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #94a3b8 !important;
    }


    /* ========================================================
       SELECT BOX
    ======================================================== */

    [data-baseweb="select"] {
        background: #102a40 !important;
    }

    [data-baseweb="select"] * {
        color: #ffffff !important;
    }


    /* ========================================================
       SLIDER
    ======================================================== */

    [data-testid="stSlider"] label {
        color: #f8fafc !important;
    }

    [data-testid="stSlider"] p {
        color: #cbd5e1 !important;
    }


    /* ========================================================
       ALERTS
    ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 13px !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #f8fafc !important;
    }


    /* ========================================================
       PROGRESS
    ======================================================== */

    [data-testid="stProgressBar"] {
        background: #1e293b !important;
    }


    /* ========================================================
       PREDICTION CARD
    ======================================================== */

    .prediction-card {
        padding: 28px;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                #08324a,
                #0a2034
            );

        border: 2px solid #22d3ee;

        text-align: center;

        box-shadow:
            0 0 30px rgba(34, 211, 238, 0.12);
    }

    .prediction-title {
        color: #67e8f9 !important;

        font-size: 25px;

        font-weight: 800;
    }

    .confidence {
        color: #38bdf8 !important;

        font-size: 40px;

        font-weight: 900;

        margin-top: 8px;
    }

    .confidence-label {
        color: #cbd5e1 !important;

        font-size: 14px;
    }


    /* ========================================================
       GUIDANCE CARD
    ======================================================== */

    .guidance-card {
        padding: 20px;

        margin-top: 15px;

        border-radius: 16px;

        background: #102d42;

        border-left: 5px solid #22d3ee;

        color: #f8fafc !important;
    }

    .guidance-title {
        color: #67e8f9 !important;

        font-size: 18px;

        font-weight: 800;
    }

    .guidance-text {
        color: #e2e8f0 !important;

        font-size: 16px;
    }


    /* ========================================================
       CAMERA INFO CARD
    ======================================================== */

    .camera-card {
        padding: 18px;

        border-radius: 16px;

        background: #0c2438;

        border: 1px solid #286789;
    }

    .camera-title {
        color: #67e8f9 !important;

        font-weight: 800;

        font-size: 18px;
    }

    .camera-description {
        color: #cbd5e1 !important;

        font-size: 14px;
    }


    /* ========================================================
       FEATURE CARDS
    ======================================================== */

    .feature-card {
        padding: 20px;

        min-height: 130px;

        border-radius: 17px;

        background: #102a40;

        border: 1px solid #245b7a;
    }

    .feature-title {
        color: #67e8f9 !important;

        font-weight: 800;

        font-size: 18px;
    }

    .feature-text {
        color: #cbd5e1 !important;

        font-size: 14px;
    }


    /* ========================================================
       DIVIDER
    ======================================================== */

    hr {
        border-color: #245b7a !important;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {
        text-align: center;

        color: #94a3b8 !important;

        padding: 25px 0 10px;

        font-size: 13px;
    }

</style>
""")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "traffic_sign_cnn_model.keras"
    )


model = load_model()


# ============================================================
# CLASS NAMES
# ============================================================

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


# ============================================================
# DRIVING GUIDANCE
# ============================================================

guidance = {

    "Stop":
        "Come to a complete stop before proceeding.",

    "Yield":
        "Slow down and give priority to other road users.",

    "No entry":
        "Do not enter this road or area.",

    "Speed limit (20 km/h)":
        "Keep your speed at or below 20 km/h.",

    "Speed limit (30 km/h)":
        "Keep your speed at or below 30 km/h.",

    "Speed limit (50 km/h)":
        "Keep your speed at or below 50 km/h.",

    "Speed limit (60 km/h)":
        "Keep your speed at or below 60 km/h.",

    "Speed limit (70 km/h)":
        "Keep your speed at or below 70 km/h.",

    "Speed limit (80 km/h)":
        "Keep your speed at or below 80 km/h.",

    "Speed limit (100 km/h)":
        "Keep your speed at or below 100 km/h.",

    "Speed limit (120 km/h)":
        "Keep your speed at or below 120 km/h.",

    "No passing":
        "Overtaking is not allowed in this area.",

    "Keep right":
        "Keep to the right side of the road.",

    "Keep left":
        "Keep to the left side of the road.",

    "Turn right ahead":
        "Prepare to turn right ahead.",

    "Turn left ahead":
        "Prepare to turn left ahead.",

    "Ahead only":
        "Continue straight ahead.",

    "Roundabout mandatory":
        "Follow the roundabout direction.",

    "Traffic signals":
        "Watch for traffic lights and follow their signals.",

    "Road work":
        "Slow down and watch for road construction.",

    "Pedestrians":
        "Watch carefully for pedestrians.",

    "Children crossing":
        "Slow down and watch for children crossing.",

    "Dangerous curve left":
        "Reduce speed and prepare for a left curve.",

    "Dangerous curve right":
        "Reduce speed and prepare for a right curve.",

    "Slippery road":
        "Drive carefully because the road may be slippery.",

    "Beware of ice/snow":
        "Reduce speed and drive carefully in icy or snowy conditions.",

    "Wild animals crossing":
        "Watch for animals crossing the road.",

    "General caution":
        "Drive carefully and stay alert to road conditions."

}


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:

    st.session_state.prediction_done = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🚦 TrafficSign AI")

    st.caption(
        "CNN Based Traffic Sign Classification"
    )

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

    st.success(
        "🤖 Model Loaded\n\n"
        "43 Traffic Classes\n\n"
        "96.44% Test Accuracy"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            🚦 TrafficSign AI
        </div>

        <div class="hero-subtitle">
            Intelligent Traffic Sign Classification
            using Convolutional Neural Networks
        </div>

    </div>
    """)

    st.subheader("📊 Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Traffic Classes",
            "43"
        )

    with c2:
        st.metric(
            "Training Images",
            "39,209"
        )

    with c3:
        st.metric(
            "Test Images",
            "12,630"
        )

    with c4:
        st.metric(
            "Test Accuracy",
            "96.44%"
        )

    st.divider()

    st.subheader("🧠 CNN Architecture")

    a1, a2, a3, a4, a5 = st.columns(5)

    with a1:
        st.info(
            "**Layer 1**\n\n"
            "Conv2D\n\n"
            "32 Filters"
        )

    with a2:
        st.info(
            "**Layer 2**\n\n"
            "Conv2D\n\n"
            "64 Filters"
        )

    with a3:
        st.info(
            "**Layer 3**\n\n"
            "Conv2D\n\n"
            "128 Filters"
        )

    with a4:
        st.info(
            "**Layer 4**\n\n"
            "Dense\n\n"
            "128 Neurons"
        )

    with a5:
        st.info(
            "**Output**\n\n"
            "Softmax\n\n"
            "43 Classes"
        )

    st.divider()

    st.subheader("✨ Application Features")

    f1, f2, f3 = st.columns(3)

    with f1:

        st.html("""
        <div class="feature-card">

            <div class="feature-title">
                📷 Camera Prediction
            </div>

            <div class="feature-text">
                Capture a traffic sign directly
                using your camera.
            </div>

        </div>
        """)

    with f2:

        st.html("""
        <div class="feature-card">

            <div class="feature-title">
                📁 Image Upload
            </div>

            <div class="feature-text">
                Upload JPG, JPEG or PNG
                traffic sign images.
            </div>

        </div>
        """)

    with f3:

        st.html("""
        <div class="feature-card">

            <div class="feature-title">
                🎯 Smart Prediction
            </div>

            <div class="feature-text">
                Get confidence score and
                top 3 predictions.
            </div>

        </div>
        """)


# ============================================================
# PREDICT SIGN
# ============================================================

elif page == "🔍 Predict Sign":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            🔍 Predict Traffic Sign
        </div>

        <div class="hero-subtitle">
            Capture a traffic sign with your camera
            or upload an image for AI classification.
        </div>

    </div>
    """)

    st.subheader("📥 Select Input Method")

    input_method = st.radio(
        "How do you want to provide the image?",
        [
            "📷 Open Camera",
            "📁 Upload File"
        ],
        horizontal=True
    )

    image = None

    # ========================================================
    # CAMERA
    # ========================================================

    if input_method == "📷 Open Camera":

        st.html("""
        <div class="camera-card">

            <div class="camera-title">
                📷 Camera Mode
            </div>

            <div class="camera-description">
                Allow camera permission and capture
                a clear traffic sign.
            </div>

        </div>
        """)

        camera_image = st.camera_input(
            "Take a picture of the traffic sign"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            ).convert("RGB")


    # ========================================================
    # FILE UPLOAD
    # ========================================================

    else:

        uploaded_file = st.file_uploader(
            "📁 Upload Traffic Sign Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            help="Upload a clear traffic sign image."
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


    # ========================================================
    # IMAGE AVAILABLE
    # ========================================================

    if image is not None:

        st.divider()

        left, right = st.columns(
            [1, 1.1]
        )

        # ====================================================
        # IMAGE PREVIEW
        # ====================================================

        with left:

            st.subheader("🖼️ Image Preview")

            st.image(
                image,
                caption="Input Traffic Sign",
                use_container_width=True
            )

        # ====================================================
        # SETTINGS
        # ====================================================

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
                f"Current threshold: {threshold:.0%}"
            )

            st.write("")

            predict_button = st.button(
                "🚀 Predict Traffic Sign",
                type="primary",
                use_container_width=True
            )

            if predict_button:

                # --------------------------------------------
                # PREPROCESS IMAGE
                # --------------------------------------------

                resized_image = image.resize(
                    (32, 32)
                )

                img_array = np.array(
                    resized_image
                ).astype("float32")

                img_array = (
                    img_array / 255.0
                )

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                predictions = model.predict(
                    img_array,
                    verbose=0
                )[0]

                # --------------------------------------------
                # TOP 3
                # --------------------------------------------

                top_indices = np.argsort(
                    predictions
                )[::-1][:3]

                predicted_index = top_indices[0]

                predicted_name = class_names[
                    predicted_index
                ]

                confidence = float(
                    predictions[
                        predicted_index
                    ]
                )

                # --------------------------------------------
                # SAVE RESULT
                # --------------------------------------------

                st.session_state.prediction_done = True

                st.session_state.predicted_name = (
                    predicted_name
                )

                st.session_state.confidence = (
                    confidence
                )

                st.session_state.top_indices = (
                    top_indices
                )

                st.session_state.predictions = (
                    predictions
                )


        # ====================================================
        # RESULT
        # ====================================================

        if st.session_state.prediction_done:

            predicted_name = (
                st.session_state.predicted_name
            )

            confidence = (
                st.session_state.confidence
            )

            top_indices = (
                st.session_state.top_indices
            )

            predictions = (
                st.session_state.predictions
            )

            st.divider()

            st.subheader("🎯 Prediction Result")

            result1, result2 = st.columns(
                [1.4, 1]
            )

            # =================================================
            # PREDICTION CARD
            # =================================================

            with result1:

                st.html(f"""
                <div class="prediction-card">

                    <div class="prediction-title">
                        🚦 {predicted_name}
                    </div>

                    <div class="confidence">
                        {confidence:.2%}
                    </div>

                    <div class="confidence-label">
                        Model Confidence
                    </div>

                </div>
                """)

            # =================================================
            # CONFIDENCE STATUS
            # =================================================

            with result2:

                if confidence >= threshold:

                    st.success(
                        "✅ High Confidence\n\n"
                        "The model is confident "
                        "about this prediction."
                    )

                else:

                    st.warning(
                        "⚠️ Low Confidence\n\n"
                        "Try a clearer or closer "
                        "traffic sign image."
                    )

            # =================================================
            # CONFIDENCE PROGRESS
            # =================================================

            st.write("")

            st.write(
                f"**Confidence Level: {confidence:.2%}**"
            )

            st.progress(
                confidence
            )

            # =================================================
            # DRIVING GUIDANCE
            # =================================================

            st.subheader("🚘 Driving Guidance")

            instruction = guidance.get(
                predicted_name,
                "Follow the traffic rules associated with this sign."
            )

            st.html(f"""
            <div class="guidance-card">

                <div class="guidance-title">
                    💡 Recommended Action
                </div>

                <br>

                <div class="guidance-text">
                    {instruction}
                </div>

            </div>
            """)

            # =================================================
            # TOP 3 PREDICTIONS
            # =================================================

            st.divider()

            st.subheader(
                "🏆 Top 3 Predictions"
            )

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                name = class_names[index]

                score = float(
                    predictions[index]
                )

                col1, col2 = st.columns(
                    [2, 4]
                )

                with col1:

                    st.write(
                        f"**#{rank}  {name}**"
                    )

                with col2:

                    st.progress(
                        min(score, 1.0),
                        text=f"{score:.2%}"
                    )

            # =================================================
            # RESET
            # =================================================

            st.divider()

            if st.button(
                "🔄 Clear Prediction",
                use_container_width=True
            ):

                st.session_state.prediction_done = False

                st.rerun()

    else:

        st.info(
            "👆 Select **Open Camera** or "
            "**Upload File** above to start prediction."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            📊 Model Performance
        </div>

        <div class="hero-subtitle">
            Evaluation results of the trained CNN model.
        </div>

    </div>
    """)

    st.subheader("📈 Test Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Accuracy",
            "96.44%"
        )

    with c2:
        st.metric(
            "Error Rate",
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

    st.subheader("⚙️ Model Configuration")

    st.info(
        """
        **Input Size:** 32 × 32 × 3

        **Architecture:** Convolutional Neural Network

        **Optimizer:** Adam

        **Loss Function:** Sparse Categorical Crossentropy

        **Epochs:** 15

        **Batch Size:** 64

        **Output Classes:** 43
        """
    )

    st.divider()

    st.subheader("🔄 Prediction Pipeline")

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.info("📷\n\n**Input Image**")

    with p2:
        st.info("📐\n\n**Resize 32×32**")

    with p3:
        st.info("⚖️\n\n**Normalize**")

    with p4:
        st.info("🧠\n\n**CNN Model**")

    with p5:
        st.info("🎯\n\n**Prediction**")


# ============================================================
# SIGN CLASSES
# ============================================================

elif page == "🚦 Sign Classes":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            🚦 Traffic Sign Classes
        </div>

        <div class="hero-subtitle">
            Explore all 43 traffic sign categories
            recognized by the CNN model.
        </div>

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
        f"Showing **{len(filtered_classes)}** "
        f"of **{len(class_names)}** classes"
    )

    st.divider()

    for index, name in filtered_classes:

        col1, col2 = st.columns(
            [1, 5]
        )

        with col1:

            st.info(
                f"Class {index}"
            )

        with col2:

            st.write(
                f"🚦 **{name}**"
            )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            ℹ️ About TrafficSign AI
        </div>

        <div class="hero-subtitle">
            Deep Learning based traffic sign
            classification system.
        </div>

    </div>
    """)

    st.subheader("🎓 Project Objective")

    st.write(
        """
        The objective of this project is to develop
        a Deep Learning model that can automatically
        recognize and classify traffic signs from images.
        """
    )

    st.divider()

    st.subheader("🧠 Technology Stack")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.info("🐍 Python")

    with t2:
        st.info("🧠 TensorFlow")

    with t3:
        st.info("🔬 CNN")

    with t4:
        st.info("🌐 Streamlit")

    st.divider()

    st.subheader("📊 Dataset")

    st.write(
        """
        The dataset contains **43 different traffic
        sign classes** used for training and evaluating
        the CNN classification model.
        """
    )

    st.divider()

    st.subheader("🚀 Application")

    st.success(
        """
        The application supports real-time traffic sign
        prediction through camera capture and image upload.

        Users can also view the confidence score,
        top 3 predictions and basic driving guidance.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        🚦 TrafficSign AI &nbsp; | &nbsp;
        CNN Based Traffic Sign Classification
        &nbsp; | &nbsp;
        Deep Learning Final Year Project
    </div>
    """,
    unsafe_allow_html=True
)
