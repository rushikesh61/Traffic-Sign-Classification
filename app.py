```python
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
# PROFESSIONAL DARK UI
# =========================================================

st.html("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 30%),
        radial-gradient(circle at bottom left, rgba(37,99,235,0.08), transparent 30%),
        #071522;
    color: #f8fafc;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #081a2a;
    border-right: 1px solid #1e3a52;
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* Sidebar title */
.sidebar-title {
    font-size: 25px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 2px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Hero */
.hero {
    background:
        linear-gradient(135deg, #0b253b, #0b1c2c);
    border: 1px solid #1d4f70;
    border-radius: 24px;
    padding: 35px;
    margin-bottom: 25px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 42px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 8px;
}

.hero-title span {
    color: #38bdf8;
}

.hero-subtitle {
    font-size: 17px;
    color: #94a3b8;
    line-height: 1.6;
}

/* Section title */
.section-title {
    font-size: 26px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Cards */
.card {
    background: #0c2235;
    border: 1px solid #1c425c;
    border-radius: 18px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.18);
}

.card:hover {
    border-color: #38bdf8;
}

.card-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

.card-title {
    font-size: 18px;
    font-weight: 750;
    color: #f8fafc;
    margin-bottom: 7px;
}

.card-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.5;
}

/* KPI cards */
.kpi {
    background: #0c2235;
    border: 1px solid #1c425c;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.kpi-value {
    font-size: 29px;
    font-weight: 900;
    color: #38bdf8;
}

.kpi-label {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 5px;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, #0c3047, #0b2235);
    border: 2px solid #38bdf8;
    border-radius: 22px;
    padding: 28px;
    margin-top: 20px;
    text-align: center;
}

.prediction-label {
    font-size: 14px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-name {
    font-size: 30px;
    font-weight: 900;
    color: #38bdf8;
    margin: 8px 0;
}

.prediction-confidence {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
}

/* Camera / upload card */
.camera-card {
    background: #102a40;
    border: 2px solid #38bdf8;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 12px;
}

.camera-title {
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
}

.camera-description {
    font-size: 14px;
    color: #cbd5e1;
    margin-top: 6px;
    line-height: 1.5;
}

/* CAMERA INPUT */
[data-testid="stCameraInput"] {
    background: #102a40 !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 18px !important;
    padding: 18px !important;
}

/* Camera buttons */
[data-testid="stCameraInput"] button {
    background: #0284c7 !important;
    color: #ffffff !important;
    border: 1px solid #67e8f9 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

[data-testid="stCameraInput"] button:hover {
    background: #0369a1 !important;
    color: #ffffff !important;
}

[data-testid="stCameraInput"] button span {
    color: #ffffff !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #102a40 !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 18px !important;
    padding: 18px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0b2439 !important;
    border: 2px dashed #38bdf8 !important;
    border-radius: 14px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #f8fafc !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #0284c7 !important;
    color: #ffffff !important;
    border: 1px solid #67e8f9 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #0369a1 !important;
    color: #ffffff !important;
}

[data-testid="stFileUploaderDropzone"] button span {
    color: #ffffff !important;
}

/* All normal buttons */
.stButton > button {
    background: #0284c7 !important;
    color: white !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 45px !important;
}

.stButton > button:hover {
    background: #0369a1 !important;
    border-color: #67e8f9 !important;
}

/* Inputs */
.stTextInput input,
.stSelectbox,
.stRadio,
.stSlider {
    color: #f8fafc !important;
}

/* Progress */
.stProgress > div > div > div {
    background-color: #38bdf8 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #0c2235 !important;
    color: #f8fafc !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #1c425c;
    border-radius: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #1e3a52;
}

</style>
""")

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_cnn_model.keras")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    st.error("Model could not be loaded.")
    st.code(str(e))


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

    "Speed limit (20 km/h)": "Maintain speed at or below 20 km/h.",
    "Speed limit (30 km/h)": "Maintain speed at or below 30 km/h.",
    "Speed limit (50 km/h)": "Maintain speed at or below 50 km/h.",
    "Speed limit (60 km/h)": "Maintain speed at or below 60 km/h.",
    "Speed limit (70 km/h)": "Maintain speed at or below 70 km/h.",
    "Speed limit (80 km/h)": "Maintain speed at or below 80 km/h.",
    "Speed limit (100 km/h)": "Maintain speed at or below 100 km/h.",
    "Speed limit (120 km/h)": "Maintain speed at or below 120 km/h.",

    "Stop": "Come to a complete stop and check the road before proceeding.",
    "Yield": "Slow down and give priority to other road users.",
    "No entry": "Do not enter this road from this direction.",
    "No passing": "Overtaking is not allowed in this section.",
    "Keep right": "Keep your vehicle on the right side of the road.",
    "Keep left": "Keep your vehicle on the left side of the road.",
    "Roundabout mandatory": "Follow the roundabout direction.",
    "Road work": "Slow down and watch for road construction.",
    "Traffic signals": "Follow the traffic signal.",
    "Pedestrians": "Watch carefully for pedestrians and reduce speed.",
    "Children crossing": "Slow down and watch for children.",
    "Bicycles crossing": "Watch for cyclists crossing the road.",
    "Slippery road": "Reduce speed because the road may be slippery.",
    "Dangerous curve left": "Slow down and prepare for a left curve.",
    "Dangerous curve right": "Slow down and prepare for a right curve.",
    "Wild animals crossing": "Watch for animals crossing the road.",
    "Beware of ice/snow": "Drive carefully because ice or snow may be present.",
    "General caution": "Drive carefully and watch for possible hazards."
}


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🚦 TrafficSign AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Traffic Sign Classification using CNN'
        '</div>',
        unsafe_allow_html=True
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

    if model_loaded:
        st.success("🟢 CNN Model Loaded")
    else:
        st.error("🔴 Model Not Loaded")

    st.caption("Deep Learning Project")
    st.caption("43 Traffic Sign Classes")


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            🚦 Traffic<span>Sign AI</span>
        </div>

        <div class="hero-subtitle">
            Intelligent traffic sign classification using
            Convolutional Neural Networks (CNN).
            Upload an image or capture a traffic sign
            using your camera and let the AI identify it.
        </div>

    </div>
    """)

    st.markdown(
        '<div class="section-title">📊 Project Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">43</div>
            <div class="kpi-label">Traffic Sign Classes</div>
        </div>
        """)

    with c2:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">39,209</div>
            <div class="kpi-label">Training Images</div>
        </div>
        """)

    with c3:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">12,630</div>
            <div class="kpi-label">Test Images</div>
        </div>
        """)

    with c4:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">96.44%</div>
            <div class="kpi-label">Test Accuracy</div>
        </div>
        """)

    st.markdown(
        '<div class="section-title">🧠 CNN Architecture</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.html("""
        <div class="card">
            <div class="card-icon">🧠</div>
            <div class="card-title">Convolution Layers</div>
            <div class="card-text">
                3 Conv2D layers using 32, 64 and 128 filters
                to learn visual patterns from traffic signs.
            </div>
        </div>
        """)

    with c2:
        st.html("""
        <div class="card">
            <div class="card-icon">⚡</div>
            <div class="card-title">Feature Learning</div>
            <div class="card-text">
                MaxPooling and ReLU activation help extract
                important shape, color and edge features.
            </div>
        </div>
        """)

    with c3:
        st.html("""
        <div class="card">
            <div class="card-icon">🎯</div>
            <div class="card-title">Classification</div>
            <div class="card-text">
                Dense layers and Softmax classify the image
                into one of 43 traffic sign categories.
            </div>
        </div>
        """)

    st.markdown(
        '<div class="section-title">✨ Application Features</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.html("""
        <div class="card">
            <div class="card-icon">📷</div>
            <div class="card-title">Camera Prediction</div>
            <div class="card-text">
                Capture a traffic sign directly using
                your device camera.
            </div>
        </div>
        """)

    with c2:
        st.html("""
        <div class="card">
            <div class="card-icon">📁</div>
            <div class="card-title">Image Upload</div>
            <div class="card-text">
                Upload JPG, JPEG or PNG traffic sign images
                for classification.
            </div>
        </div>
        """)

    with c3:
        st.html("""
        <div class="card">
            <div class="card-icon">🤖</div>
            <div class="card-title">AI Prediction</div>
            <div class="card-text">
                CNN provides the predicted sign, confidence
                score and top alternative predictions.
            </div>
        </div>
        """)


# =========================================================
# PREDICT SIGN
# =========================================================

elif page == "🔍 Predict Sign":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            🔍 Predict <span>Traffic Sign</span>
        </div>

        <div class="hero-subtitle">
            Capture a traffic sign using your camera or
            upload an image. The trained CNN model will
            classify the sign.
        </div>

    </div>
    """)

    # Input method
    input_method = st.radio(
        "Select Input Method",
        [
            "📷 Open Camera",
            "📁 Upload File"
        ],
        horizontal=True
    )

    image = None

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    if input_method == "📷 Open Camera":

        st.html("""
        <div class="camera-card">

            <div class="camera-title">
                📷 Camera Mode
            </div>

            <div class="camera-description">
                Take a clear photo of the traffic sign.
                Allow camera permission when requested.
            </div>

        </div>
        """)

        camera_image = st.camera_input(
            "📷 Take Photo"
        )

        if camera_image is not None:
            image = Image.open(
                camera_image
            ).convert("RGB")

    # -----------------------------------------------------
    # FILE UPLOAD
    # -----------------------------------------------------

    else:

        st.html("""
        <div class="camera-card">

            <div class="camera-title">
                📁 Upload Image
            </div>

            <div class="camera-description">
                Choose a JPG, JPEG or PNG image
                containing a traffic sign.
            </div>

        </div>
        """)

        uploaded_file = st.file_uploader(
            "📁 Browse Files",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            label_visibility="visible"
        )

        if uploaded_file is not None:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    if image is not None:

        st.markdown(
            '<div class="section-title">🖼️ Image Preview</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            st.image(
                image,
                caption="Input Traffic Sign",
                use_container_width=True
            )

        with c2:

            st.markdown(
                '<div class="section-title">⚙️ Prediction Settings</div>',
                unsafe_allow_html=True
            )

            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.70,
                step=0.01
            )

            st.info(
                f"Prediction will be considered reliable "
                f"when confidence is ≥ "
                f"{confidence_threshold:.0%}."
            )

            predict_button = st.button(
                "🚀 Predict Traffic Sign",
                use_container_width=True
            )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        if predict_button:

            if not model_loaded:
                st.error(
                    "Model is not available."
                )

            else:

                with st.spinner(
                    "🤖 AI is analyzing the image..."
                ):

                    # Resize
                    img = image.resize(
                        (32, 32)
                    )

                    # Convert to numpy
                    img_array = np.array(
                        img,
                        dtype=np.float32
                    )

                    # Normalize
                    img_array = img_array / 255.0

                    # Add batch dimension
                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    # Prediction
                    predictions = model.predict(
                        img_array,
                        verbose=0
                    )[0]

                    predicted_class = int(
                        np.argmax(predictions)
                    )

                    confidence = float(
                        predictions[predicted_class]
                    )

                    # Top 3
                    top_indices = np.argsort(
                        predictions
                    )[-3:][::-1]

                    st.session_state.prediction_result = {
                        "class": class_names[predicted_class],
                        "confidence": confidence,
                        "top_indices": top_indices,
                        "predictions": predictions,
                        "threshold": confidence_threshold
                    }

                    st.session_state.prediction_done = True

    # -----------------------------------------------------
    # SHOW RESULT
    # -----------------------------------------------------

    if (
        st.session_state.prediction_done
        and st.session_state.prediction_result
    ):

        result = st.session_state.prediction_result

        sign_name = result["class"]
        confidence = result["confidence"]
        threshold = result["threshold"]

        st.markdown(
            '<div class="section-title">🎯 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.html(f"""
        <div class="prediction-card">

            <div class="prediction-label">
                Detected Traffic Sign
            </div>

            <div class="prediction-name">
                {sign_name}
            </div>

            <div class="prediction-confidence">
                Confidence: {confidence:.2%}
            </div>

        </div>
        """)

        st.write("")

        # Confidence status
        if confidence >= threshold:

            st.success(
                f"✅ High confidence prediction: "
                f"{confidence:.2%}"
            )

        else:

            st.warning(
                f"⚠️ Low confidence prediction: "
                f"{confidence:.2%}. "
                f"Try a clearer image."
            )

        st.progress(
            confidence
        )

        # Guidance
        if sign_name in guidance:

            st.markdown(
                '<div class="section-title">🚗 Driving Guidance</div>',
                unsafe_allow_html=True
            )

            st.info(
                guidance[sign_name]
            )

        # Top 3 predictions
        st.markdown(
            '<div class="section-title">🏆 Top 3 Predictions</div>',
            unsafe_allow_html=True
        )

        top_indices = result["top_indices"]
        predictions = result["predictions"]

        for rank, idx in enumerate(top_indices, start=1):

            col1, col2, col3 = st.columns(
                [1, 6, 2]
            )

            with col1:
                st.write(
                    f"**#{rank}**"
                )

            with col2:
                st.write(
                    class_names[int(idx)]
                )

            with col3:
                st.write(
                    f"{predictions[int(idx)]:.2%}"
                )

        st.write("")

        if st.button(
            "🗑️ Clear Prediction",
            use_container_width=True
        ):

            st.session_state.prediction_done = False
            st.session_state.prediction_result = None

            st.rerun()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            📊 Model <span>Performance</span>
        </div>

        <div class="hero-subtitle">
            Evaluation results of the custom CNN model
            on the GTSRB traffic sign test dataset.
        </div>

    </div>
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">96.44%</div>
            <div class="kpi-label">Test Accuracy</div>
        </div>
        """)

    with c2:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">94.41%</div>
            <div class="kpi-label">Macro F1 Score</div>
        </div>
        """)

    with c3:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">96.41%</div>
            <div class="kpi-label">Weighted F1</div>
        </div>
        """)

    with c4:
        st.html("""
        <div class="kpi">
            <div class="kpi-value">3.56%</div>
            <div class="kpi-label">Test Error</div>
        </div>
        """)

    st.markdown(
        '<div class="section-title">📈 Training Information</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.html("""
        <div class="card">

            <div class="card-title">
                Training Dataset
            </div>

            <div class="card-text">

                Training Images: 39,209<br>
                Validation Split: 20%<br>
                Validation Images: 7,842<br>
                Test Images: 12,630<br>
                Number of Classes: 43

            </div>

        </div>
        """)

    with c2:

        st.html("""
        <div class="card">

            <div class="card-title">
                CNN Configuration
            </div>

            <div class="card-text">

                Input Size: 32 × 32 × 3<br>
                Conv Filters: 32 → 64 → 128<br>
                Dense Layer: 128 neurons<br>
                Dropout: 0.5<br>
                Optimizer: Adam<br>
                Loss: Sparse Categorical Crossentropy

            </div>

        </div>
        """)

    st.markdown(
        '<div class="section-title">🔬 Test Results</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The CNN achieved **96.44% accuracy** on the "
        "12,630-image test dataset."
    )

    st.write(
        "Correct Predictions: **12,180**"
    )

    st.write(
        "Incorrect Predictions: **450**"
    )

    st.write(
        "Test Error Rate: **3.56%**"
    )

    st.info(
        "The model performs well overall, although visually "
        "similar signs such as speed-limit signs and some "
        "warning signs can be more difficult to classify."
    )


# =========================================================
# SIGN CLASSES
# =========================================================

elif page == "🚦 Sign Classes":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            🚦 Traffic Sign <span>Classes</span>
        </div>

        <div class="hero-subtitle">
            The CNN model can classify 43 different
            traffic sign categories.
        </div>

    </div>
    """)

    search = st.text_input(
        "🔎 Search Traffic Sign",
        placeholder="Example: speed, stop, road, curve..."
    )

    filtered_classes = []

    for i, name in enumerate(class_names):

        if search.lower() in name.lower():

            filtered_classes.append(
                (i, name)
            )

    st.write(
        f"Showing **{len(filtered_classes)}** "
        f"of **43** classes."
    )

    for start in range(
        0,
        len(filtered_classes),
        3
    ):

        cols = st.columns(3)

        for col, item in zip(
            cols,
            filtered_classes[start:start + 3]
        ):

            class_id, name = item

            with col:

                st.html(f"""
                <div class="card">

                    <div class="card-icon">
                        🚦
                    </div>

                    <div class="card-title">
                        Class {class_id}
                    </div>

                    <div class="card-text">
                        {name}
                    </div>

                </div>
                """)

                st.write("")


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.html("""
    <div class="hero">

        <div class="hero-title">
            ℹ️ About <span>TrafficSign AI</span>
        </div>

        <div class="hero-subtitle">
            Deep Learning based Traffic Sign Classification
            project using Convolutional Neural Networks.
        </div>

    </div>
    """)

    st.markdown(
        '<div class="section-title">🎯 Project Objective</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The objective of this project is to develop a deep
        learning system capable of automatically identifying
        traffic signs from images.

        The project uses a Convolutional Neural Network (CNN)
        to learn visual features such as edges, shapes,
        colors and patterns from traffic sign images.
        """
    )

    st.markdown(
        '<div class="section-title">🛠️ Technologies Used</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.html("""
        <div class="card">
            <div class="card-icon">🐍</div>
            <div class="card-title">Python</div>
            <div class="card-text">
                Main programming language used
                for data processing and model development.
            </div>
        </div>
        """)

    with c2:
        st.html("""
        <div class="card">
            <div class="card-icon">🧠</div>
            <div class="card-title">TensorFlow / Keras</div>
            <div class="card-text">
                Used to build, train and evaluate
                the convolutional neural network.
            </div>
        </div>
        """)

    with c3:
        st.html("""
        <div class="card">
            <div class="card-icon">🎨</div>
            <div class="card-title">Streamlit</div>
            <div class="card-text">
                Used to create the interactive web
                application for real-time prediction.
            </div>
        </div>
        """)

    st.markdown(
        '<div class="section-title">🔄 Project Workflow</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        **1. Dataset Collection**
        
        Traffic sign images are collected from the
        GTSRB-style dataset.

        **2. Data Preprocessing**

        Images are resized to 32 × 32 pixels and pixel
        values are normalized between 0 and 1.

        **3. Model Training**

        A CNN is trained using convolution, pooling,
        dense and dropout layers.

        **4. Model Evaluation**

        The trained model is evaluated using test
        accuracy and classification metrics.

        **5. Deployment**

        The trained Keras model is integrated into
        a Streamlit application.

        **6. Prediction**

        Users can upload an image or take a photo
        and receive the predicted traffic sign.
        """
    )

    st.markdown(
        '<div class="section-title">⚠️ Important Note</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "This application is a college deep learning "
        "project and should not be used as a certified "
        "vehicle safety or autonomous-driving system."
    )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    🚦 TrafficSign AI |
    CNN Based Traffic Sign Classification |
    Deep Learning Project

    <br><br>

    Built with Python • TensorFlow • Keras • Streamlit

</div>
""")
```
