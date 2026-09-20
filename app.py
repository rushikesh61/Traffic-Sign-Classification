import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrafficSign AI | CNN Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL CONFIG
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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 229, 255, 0.08), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(59, 130, 246, 0.10), transparent 30%),
        #07111f;
    color: #f8fafc;
}

/* Main content */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #07111f 0%, #0b1729 100%);
    border-right: 1px solid rgba(148,163,184,0.15);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Sidebar text */

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #cbd5e1 !important;
}

/* Buttons */

.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(34,211,238,0.25);
    background: linear-gradient(135deg, #0e7490, #2563eb);
    color: white;
    font-weight: 700;
    padding: 0.65rem 1rem;
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(34,211,238,0.22);
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.65);
    border: 1px dashed rgba(34,211,238,0.5);
    border-radius: 18px;
    padding: 10px;
}

/* Cards */

.card {
    background: linear-gradient(
        145deg,
        rgba(15, 23, 42, 0.96),
        rgba(15, 32, 55, 0.90)
    );
    border: 1px solid rgba(148,163,184,0.13);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.18);
    margin-bottom: 18px;
}

.kpi-card {
    background: linear-gradient(
        145deg,
        rgba(14,116,144,0.16),
        rgba(37,99,235,0.12)
    );
    border: 1px solid rgba(34,211,238,0.18);
    border-radius: 18px;
    padding: 20px;
    min-height: 135px;
}

.kpi-title {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.kpi-value {
    color: #f8fafc;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-sub {
    color: #67e8f9;
    font-size: 0.8rem;
    margin-top: 5px;
}

/* Hero */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    padding: 45px;
    margin-bottom: 28px;
    background:
        linear-gradient(120deg,
            rgba(8,47,73,0.98),
            rgba(15,23,42,0.98) 55%,
            rgba(30,58,138,0.85));
    border: 1px solid rgba(34,211,238,0.18);
    box-shadow: 0 25px 70px rgba(0,0,0,0.28);
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -90px;
    top: -120px;
    border-radius: 50%;
    background: rgba(34,211,238,0.10);
}

.hero-badge {
    display: inline-block;
    background: rgba(34,211,238,0.10);
    color: #67e8f9;
    border: 1px solid rgba(34,211,238,0.25);
    padding: 7px 13px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 15px;
}

.hero-title {
    font-size: 3.1rem;
    font-weight: 800;
    line-height: 1.05;
    color: white;
    margin-bottom: 14px;
}

.hero-title span {
    color: #22d3ee;
}

.hero-description {
    color: #cbd5e1;
    font-size: 1.02rem;
    max-width: 700px;
    line-height: 1.7;
}

.hero-visual {
    font-size: 7rem;
    text-align: center;
    padding-top: 15px;
    filter: drop-shadow(0 15px 30px rgba(34,211,238,0.2));
}

/* Section */

.section-title {
    color: #f8fafc;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 12px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* Prediction result */

.prediction-box {
    background:
        linear-gradient(135deg,
        rgba(8,47,73,0.95),
        rgba(15,23,42,0.95));
    border: 1px solid rgba(34,211,238,0.3);
    border-radius: 22px;
    padding: 28px;
    text-align: center;
}

.prediction-icon {
    font-size: 3rem;
    margin-bottom: 10px;
}

.prediction-label {
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-name {
    color: #67e8f9;
    font-size: 1.65rem;
    font-weight: 800;
    margin: 8px 0;
}

.confidence {
    color: #f8fafc;
    font-size: 2.2rem;
    font-weight: 800;
}

/* Status */

.status {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
}

.status-success {
    background: rgba(34,197,94,0.12);
    color: #86efac;
    border: 1px solid rgba(34,197,94,0.25);
}

.status-warning {
    background: rgba(245,158,11,0.12);
    color: #fcd34d;
    border: 1px solid rgba(245,158,11,0.25);
}

/* Architecture */

.arch-step {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    min-height: 100px;
}

.arch-icon {
    font-size: 1.8rem;
}

.arch-title {
    color: white;
    font-weight: 700;
    margin-top: 7px;
}

.arch-text {
    color: #94a3b8;
    font-size: 0.75rem;
    margin-top: 3px;
}

/* Footer */

.footer {
    text-align: center;
    padding: 25px 0 5px;
    color: #64748b;
    font-size: 0.78rem;
}

.footer strong {
    color: #22d3ee;
}

/* Hide Streamlit menu */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((32, 32))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_image(image):
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:
        results.append({
            "class_id": int(index),
            "name": CLASS_NAMES[int(index)],
            "confidence": float(predictions[index])
        })

    return results


def get_sign_emoji(class_id):

    if class_id in range(0, 9):
        return "🔴"

    if class_id in [13, 14]:
        return "🛑"

    if class_id in [17]:
        return "🚫"

    if class_id in [18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31]:
        return "⚠️"

    if class_id in [33, 34, 35, 36, 37, 38, 39, 40]:
        return "🔵"

    return "🚦"


def recommendation(class_id):

    recommendations = {
        0: "Reduce speed to 20 km/h and follow the posted limit.",
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
        11: "Give priority according to the intersection rules.",
        12: "You are travelling on a priority road.",
        13: "Slow down and give way to other road users.",
        14: "Stop completely before proceeding.",
        15: "Vehicles are prohibited in this area.",
        16: "Vehicles over 3.5 tons are prohibited.",
        17: "Entry is prohibited from this direction.",
        18: "Drive carefully and be alert for hazards.",
        19: "Prepare for a dangerous curve to the left.",
        20: "Prepare for a dangerous curve to the right.",
        21: "Reduce speed and prepare for consecutive curves.",
        22: "Drive carefully because the road may be uneven.",
        23: "Reduce speed because the road may be slippery.",
        24: "The road narrows on the right. Drive carefully.",
        25: "Road construction or maintenance is ahead.",
        26: "Traffic signals are ahead.",
        27: "Watch carefully for pedestrians.",
        28: "Watch carefully for children crossing.",
        29: "Watch carefully for bicycles.",
        30: "Be alert for ice or snow on the road.",
        31: "Watch for wild animals crossing the road.",
        32: "All previous speed and passing restrictions have ended.",
        33: "Turn right ahead.",
        34: "Turn left ahead.",
        35: "Continue straight ahead.",
        36: "You may continue straight or turn right.",
        37: "You may continue straight or turn left.",
        38: "Keep to the right.",
        39: "Keep to the left.",
        40: "Follow the roundabout direction.",
        41: "The no-passing restriction has ended.",
        42: "The heavy-vehicle no-passing restriction has ended."
    }

    return recommendations.get(
        class_id,
        "Follow the road signs and drive carefully."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px;">

        <div style="
            width:72px;
            height:72px;
            margin:auto;
            border-radius:20px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:linear-gradient(135deg,#0891b2,#2563eb);
            font-size:38px;
            box-shadow:0 10px 30px rgba(34,211,238,0.2);
        ">
            🚦
        </div>

        <h2 style="
            color:white;
            margin:12px 0 2px;
            font-size:1.25rem;
        ">
            TrafficSign AI
        </h2>

        <p style="
            color:#67e8f9;
            font-size:0.75rem;
            margin:0;
        ">
            CNN CLASSIFICATION SYSTEM
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Predict Sign",
            "📊 Model Performance",
            "🚦 Sign Classes",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    if model_loaded:
        st.markdown(
            '<span class="status status-success">● MODEL ONLINE</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status status-warning">● MODEL ERROR</span>',
            unsafe_allow_html=True
        )

    st.markdown("""
    <div style="
        margin-top:20px;
        color:#64748b;
        font-size:0.72rem;
        line-height:1.6;
    ">
        <b style="color:#94a3b8;">Technology</b><br>
        TensorFlow • Keras • CNN<br><br>

        <b style="color:#94a3b8;">Input</b><br>
        32 × 32 RGB Image<br><br>

        <b style="color:#94a3b8;">Classes</b><br>
        43 Traffic Sign Categories
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            🧠 DEEP LEARNING • COMPUTER VISION
        </div>

        <div style="display:flex; align-items:center;">

            <div style="flex:1;">

                <div class="hero-title">
                    Traffic Sign<br>
                    <span>Classification AI</span>
                </div>

                <div class="hero-description">
                    An intelligent image classification system powered by
                    Convolutional Neural Networks. Upload a traffic sign
                    image and let the trained CNN identify the sign category
                    with confidence.
                </div>

            </div>

            <div class="hero-visual">
                🚦
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Key information about the trained deep learning system</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Traffic Classes</div>
            <div class="kpi-value">43</div>
            <div class="kpi-sub">Sign Categories</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Test Images</div>
            <div class="kpi-value">12,630</div>
            <div class="kpi-sub">Evaluation Dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Test Accuracy</div>
            <div class="kpi-value">96.44%</div>
            <div class="kpi-sub">CNN Performance</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Input Size</div>
            <div class="kpi-value">32×32</div>
            <div class="kpi-sub">RGB Image</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture

    st.markdown(
        '<div class="section-title">CNN Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">How the image moves through the deep learning model</div>',
        unsafe_allow_html=True
    )

    a1, a2, a3, a4, a5 = st.columns(5)

    architecture = [
        ("🖼️", "Input", "32×32×3"),
        ("🧩", "Conv2D", "32 Filters"),
        ("🧠", "Conv2D", "64 + 128"),
        ("🔄", "Dense", "128 Neurons"),
        ("🚦", "Output", "43 Classes")
    ]

    for col, data in zip([a1, a2, a3, a4, a5], architecture):

        with col:

            st.markdown(f"""
            <div class="arch-step">

                <div class="arch-icon">{data[0]}</div>

                <div class="arch-title">
                    {data[1]}
                </div>

                <div class="arch-text">
                    {data[2]}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset information

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                🎯 Project Objective
            </div>

            <p style="color:#cbd5e1; line-height:1.8;">
                The objective of this project is to automatically recognize
                and classify road traffic signs from images using a
                Convolutional Neural Network.
            </p>

            <p style="color:#94a3b8; line-height:1.8;">
                The system learns visual patterns such as shape, color,
                symbols and road-sign structures from training images.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                ⚙️ System Status
            </div>

            <p style="margin-top:15px;">
                <span class="status status-success">
                    ● MODEL LOADED
                </span>
            </p>

            <p style="color:#94a3b8; font-size:0.85rem;">
                CNN model is ready for image classification.
            </p>

            <p style="color:#94a3b8; font-size:0.85rem;">
                Framework: TensorFlow / Keras
            </p>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔍 Predict Sign":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            🔍 AI IMAGE ANALYSIS
        </div>

        <div class="hero-title">
            Identify a <span>Traffic Sign</span>
        </div>

        <div class="hero-description">
            Upload an image containing a road traffic sign.
            The CNN model will process the image and return
            the predicted class with confidence.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:

        st.error(
            "Model could not be loaded. Please check the model file."
        )

    else:

        left, right = st.columns([1, 1])

        with left:

            st.markdown(
                '<div class="section-title">📤 Upload Image</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">Supported formats: JPG, JPEG, PNG</div>',
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Upload a traffic sign image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            confidence_threshold = st.slider(
                "Prediction Confidence Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.70,
                step=0.01
            )

            st.caption(
                "A lower-confidence prediction will be shown as "
                "uncertain. This threshold does not create a true "
                "out-of-distribution detector."
            )

        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")

            with left:

                st.image(
                    image,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            results = predict_image(image)

            best = results[0]

            with right:

                st.markdown(
                    '<div class="section-title">🤖 AI Prediction</div>',
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="prediction-box">

                    <div class="prediction-icon">
                        {get_sign_emoji(best["class_id"])}
                    </div>

                    <div class="prediction-label">
                        Predicted Traffic Sign
                    </div>

                    <div class="prediction-name">
                        {best["name"]}
                    </div>

                    <div class="confidence">
                        {best["confidence"] * 100:.2f}%
                    </div>

                    <div style="color:#94a3b8; margin-top:5px;">
                        Confidence
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if best["confidence"] >= confidence_threshold:

                    st.success(
                        "High-confidence prediction"
                    )

                else:

                    st.warning(
                        "Low-confidence prediction — image may not "
                        "match the learned traffic-sign classes."
                    )

                st.markdown(f"""
                <div class="card">

                    <div class="section-title">
                        💡 Driving Guidance
                    </div>

                    <p style="
                        color:#cbd5e1;
                        line-height:1.7;
                        margin-top:12px;
                    ">
                        {recommendation(best["class_id"])}
                    </p>

                </div>
                """, unsafe_allow_html=True)

            # Top 3

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">📊 Top 3 Predictions</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">Most probable classes returned by the CNN</div>',
                unsafe_allow_html=True
            )

            for rank, result in enumerate(results, start=1):

                percentage = result["confidence"] * 100

                col1, col2, col3 = st.columns([0.12, 0.65, 0.23])

                with col1:
                    st.markdown(
                        f"<h3 style='color:#67e8f9;'>#{rank}</h3>",
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        f"""
                        <div style="
                            color:#f8fafc;
                            font-weight:700;
                            margin-top:8px;
                        ">
                            {get_sign_emoji(result["class_id"])}
                            {result["name"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(result["confidence"], 1.0)
                    )

                with col3:

                    st.markdown(
                        f"""
                        <div style="
                            text-align:right;
                            color:#67e8f9;
                            font-weight:800;
                            font-size:1.1rem;
                            margin-top:8px;
                        ">
                            {percentage:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:

            st.markdown("""
            <div class="card" style="text-align:center; padding:55px;">

                <div style="font-size:4rem;">
                    📷
                </div>

                <h2 style="color:white;">
                    Upload an Image to Begin
                </h2>

                <p style="
                    color:#94a3b8;
                    max-width:600px;
                    margin:auto;
                    line-height:1.7;
                ">
                    Choose a traffic-sign image from your computer.
                    The trained CNN will analyze it and display
                    the predicted traffic-sign category.
                </p>

            </div>
            """, unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            📊 MODEL EVALUATION
        </div>

        <div class="hero-title">
            CNN <span>Performance</span>
        </div>

        <div class="hero-description">
            Evaluation results obtained from the independent test dataset.
            These metrics demonstrate how accurately the trained model
            classifies unseen traffic-sign images.
        </div>

    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)

    metrics = [
        ("Test Accuracy", "96.44%", "12,180 correct"),
        ("Test Error", "3.56%", "450 incorrect"),
        ("Macro F1", "94.41%", "43 classes"),
        ("Weighted F1", "96.41%", "Test dataset")
    ]

    for col, metric in zip([p1, p2, p3, p4], metrics):

        with col:

            st.markdown(f"""
            <div class="kpi-card">

                <div class="kpi-title">
                    {metric[0]}
                </div>

                <div class="kpi-value">
                    {metric[1]}
                </div>

                <div class="kpi-sub">
                    {metric[2]}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                🧠 Model Configuration
            </div>

            <table style="
                width:100%;
                color:#cbd5e1;
                border-collapse:collapse;
                margin-top:15px;
            ">

                <tr>
                    <td style="padding:10px 0;">Architecture</td>
                    <td style="text-align:right;">
                        CNN
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">Input Shape</td>
                    <td style="text-align:right;">
                        32 × 32 × 3
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">Output Classes</td>
                    <td style="text-align:right;">
                        43
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">Optimizer</td>
                    <td style="text-align:right;">
                        Adam
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">Loss Function</td>
                    <td style="text-align:right;">
                        Sparse Categorical Crossentropy
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">Epochs</td>
                    <td style="text-align:right;">
                        15
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">Batch Size</td>
                    <td style="text-align:right;">
                        64
                    </td>
                </tr>

            </table>

        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                📈 Evaluation Summary
            </div>

            <p style="color:#94a3b8; line-height:1.8;">
                The CNN achieved <b style="color:#67e8f9;">
                96.44% test accuracy
                </b> on 12,630 unseen test images.
            </p>

            <p style="color:#94a3b8; line-height:1.8;">
                The model correctly classified approximately
                <b style="color:white;">12,180 images</b> while
                <b style="color:white;">450 images</b> were incorrectly
                classified.
            </p>

            <p style="color:#94a3b8; line-height:1.8;">
                Macro F1-score of 94.41% indicates strong performance
                across the 43 traffic-sign classes.
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

        <div class="section-title">
            🔬 CNN Processing Pipeline
        </div>

        <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
            margin-top:20px;
            flex-wrap:wrap;
        ">

            <div class="arch-step" style="flex:1;">
                🖼️<br>
                <b>Input Image</b>
                <br>
                <small style="color:#94a3b8;">RGB</small>
            </div>

            <div style="color:#22d3ee; font-size:25px;">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                🧩<br>
                <b>Convolution</b>
                <br>
                <small style="color:#94a3b8;">Feature Extraction</small>
            </div>

            <div style="color:#22d3ee; font-size:25px;">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                🔄<br>
                <b>Pooling</b>
                <br>
                <small style="color:#94a3b8;">Downsampling</small>
            </div>

            <div style="color:#22d3ee; font-size:25px;">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                🧠<br>
                <b>Dense Layer</b>
                <br>
                <small style="color:#94a3b8;">Classification</small>
            </div>

            <div style="color:#22d3ee; font-size:25px;">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                🚦<br>
                <b>Prediction</b>
                <br>
                <small style="color:#94a3b8;">43 Classes</small>
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIGN CLASSES
# ============================================================

elif page == "🚦 Sign Classes":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            🚦 CLASS DATABASE
        </div>

        <div class="hero-title">
            43 Traffic <span>Sign Classes</span>
        </div>

        <div class="hero-description">
            The trained CNN recognizes 43 different traffic-sign
            categories from the dataset.
        </div>

    </div>
    """, unsafe_allow_html=True)

    search = st.text_input(
        "🔎 Search Traffic Sign",
        placeholder="Example: speed, stop, road, curve..."
    )

    filtered = []

    for class_id, name in CLASS_NAMES.items():

        if search.lower() in name.lower():

            filtered.append((class_id, name))

    if not filtered:

        st.warning("No matching traffic-sign class found.")

    else:

        for start in range(0, len(filtered), 3):

            row = filtered[start:start + 3]

            cols = st.columns(3)

            for col, (class_id, name) in zip(cols, row):

                with col:

                    st.markdown(f"""
                    <div class="card">

                        <div style="
                            font-size:2rem;
                            margin-bottom:8px;
                        ">
                            {get_sign_emoji(class_id)}
                        </div>

                        <div style="
                            color:#67e8f9;
                            font-size:0.75rem;
                            font-weight:700;
                        ">
                            CLASS {class_id}
                        </div>

                        <div style="
                            color:white;
                            font-weight:700;
                            margin-top:7px;
                            line-height:1.4;
                        ">
                            {name}
                        </div>

                    </div>
                    """, unsafe_allow_html=True)


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            🎓 FINAL YEAR DEEP LEARNING PROJECT
        </div>

        <div class="hero-title">
            About <span>TrafficSign AI</span>
        </div>

        <div class="hero-description">
            A computer vision project designed to demonstrate the
            practical application of Convolutional Neural Networks
            for automated traffic-sign classification.
        </div>

    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                🎯 Problem Statement
            </div>

            <p style="
                color:#cbd5e1;
                line-height:1.8;
            ">
                Traffic signs provide important information to drivers
                and autonomous systems. Manual recognition can be
                difficult in complex road environments.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">
                This project uses deep learning to automatically identify
                traffic signs from images.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                💡 Proposed Solution
            </div>

            <p style="
                color:#cbd5e1;
                line-height:1.8;
            ">
                A CNN model is trained on labeled traffic-sign images.
                During prediction, the uploaded image is resized,
                normalized and passed through the neural network.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">
                The final softmax layer produces probabilities for
                all 43 traffic-sign categories.
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

        <div class="section-title">
            🛠️ Technologies Used
        </div>

        <div style="
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin-top:18px;
        ">

            <span class="status status-success">
                Python
            </span>

            <span class="status status-success">
                TensorFlow
            </span>

            <span class="status status-success">
                Keras
            </span>

            <span class="status status-success">
                CNN
            </span>

            <span class="status status-success">
                NumPy
            </span>

            <span class="status status-success">
                Pillow
            </span>

            <span class="status status-success">
                Streamlit
            </span>

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

        <div class="section-title">
            🚀 Project Workflow
        </div>

        <div style="margin-top:20px;">

            <p style="color:#cbd5e1;">
                <b style="color:#22d3ee;">01</b>
                Dataset Collection
            </p>

            <p style="color:#cbd5e1;">
                <b style="color:#22d3ee;">02</b>
                Image Preprocessing
            </p>

            <p style="color:#cbd5e1;">
                <b style="color:#22d3ee;">03</b>
                Train / Validation Split
            </p>

            <p style="color:#cbd5e1;">
                <b style="color:#22d3ee;">04</b>
                CNN Model Training
            </p>

            <p style="color:#cbd5e1;">
                <b style="color:#22d3ee;">05</b>
                Model Evaluation
            </p>

            <p style="color:#cbd5e1;">
                <b style="color:#22d3ee;">06</b>
                Streamlit Deployment
            </p>

        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <strong>TrafficSign AI</strong>
    &nbsp; • &nbsp;
    Deep Learning
    &nbsp; • &nbsp;
    CNN
    &nbsp; • &nbsp;
    Computer Vision
    &nbsp; • &nbsp;
    43 Classes
    <br><br>
    Academic Project Showcase
</div>
""", unsafe_allow_html=True)
