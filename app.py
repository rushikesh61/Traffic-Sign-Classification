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
# MODEL & CLASS CONFIG
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
        radial-gradient(
            circle at 5% 5%,
            rgba(34, 211, 238, 0.08),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 15%,
            rgba(37, 99, 235, 0.10),
            transparent 30%
        ),
        #07111f;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #06101d 0%,
        #0a1728 100%
    );
    border-right: 1px solid rgba(148,163,184,0.14);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    padding: 42px;
    margin-bottom: 28px;

    background:
        linear-gradient(
            120deg,
            rgba(8,47,73,0.98),
            rgba(15,23,42,0.98) 55%,
            rgba(30,58,138,0.90)
        );

    border: 1px solid rgba(34,211,238,0.18);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.28);
}

.hero:before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    right: -120px;
    top: -160px;
    border-radius: 50%;
    background: rgba(34,211,238,0.08);
}

.hero-badge {
    display: inline-block;

    background: rgba(34,211,238,0.10);
    color: #67e8f9;

    border: 1px solid rgba(34,211,238,0.25);

    padding: 7px 14px;
    border-radius: 50px;

    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.8px;

    margin-bottom: 17px;
}

.hero-title {
    color: white;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 16px;
}

.hero-title span {
    color: #22d3ee;
}

.hero-description {
    color: #cbd5e1;
    max-width: 720px;
    font-size: 1rem;
    line-height: 1.75;
}

.hero-traffic {
    width: 180px;
    height: 180px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    background: rgba(34,211,238,0.07);
    border: 1px solid rgba(34,211,238,0.18);

    font-size: 7rem;

    box-shadow:
        0 0 60px rgba(34,211,238,0.10);
}

/* ============================================================
   CARDS
   ============================================================ */

.card {
    background:
        linear-gradient(
            145deg,
            rgba(15,23,42,0.97),
            rgba(15,32,55,0.90)
        );

    border: 1px solid rgba(148,163,184,0.13);
    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.18);

    margin-bottom: 18px;
}

.kpi-card {
    background:
        linear-gradient(
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
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.kpi-value {
    color: #f8fafc;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-sub {
    color: #67e8f9;
    font-size: 0.78rem;
    margin-top: 5px;
}

/* ============================================================
   SECTION
   ============================================================ */

.section-title {
    color: #f8fafc;
    font-size: 1.4rem;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 0.88rem;
    margin-bottom: 18px;
}

/* ============================================================
   ARCHITECTURE
   ============================================================ */

.arch-step {
    background: rgba(15,23,42,0.82);
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

/* ============================================================
   PREDICTION
   ============================================================ */

.prediction-box {
    background:
        linear-gradient(
            135deg,
            rgba(8,47,73,0.96),
            rgba(15,23,42,0.96)
        );

    border: 1px solid rgba(34,211,238,0.28);

    border-radius: 22px;

    padding: 28px;

    text-align: center;

    box-shadow:
        0 20px 50px rgba(0,0,0,0.20);
}

.prediction-icon {
    font-size: 3.2rem;
}

.prediction-label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

.prediction-name {
    color: #67e8f9;
    font-size: 1.55rem;
    font-weight: 800;
    margin: 8px 0;
}

.confidence {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 800;
}

/* ============================================================
   STATUS
   ============================================================ */

.status-online {
    display: inline-block;

    padding: 7px 13px;

    border-radius: 50px;

    background: rgba(34,197,94,0.10);

    color: #86efac;

    border: 1px solid rgba(34,197,94,0.25);

    font-size: 0.75rem;
    font-weight: 700;
}

.status-warning {
    display: inline-block;

    padding: 7px 13px;

    border-radius: 50px;

    background: rgba(245,158,11,0.10);

    color: #fcd34d;

    border: 1px solid rgba(245,158,11,0.25);

    font-size: 0.75rem;
    font-weight: 700;
}

/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    border-radius: 10px;

    border: 1px solid rgba(34,211,238,0.25);

    background:
        linear-gradient(
            135deg,
            #0e7490,
            #2563eb
        );

    color: white;

    font-weight: 700;

    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(34,211,238,0.22);
}

/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.65);

    border: 1px dashed rgba(34,211,238,0.50);

    border-radius: 18px;

    padding: 10px;
}

/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    padding: 28px 0 5px;

    color: #64748b;

    font-size: 0.76rem;
}

.footer strong {
    color: #22d3ee;
}

/* ============================================================
   HIDE STREAMLIT DEFAULT
   ============================================================ */

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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(MODEL_PATH)


try:

    model = load_model()

    model_loaded = True

except Exception:

    model = None

    model_loaded = False


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

    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    results = []

    for index in top_indices:

        results.append({
            "class_id": int(index),
            "name": CLASS_NAMES[int(index)],
            "confidence": float(predictions[index])
        })

    return results


def get_sign_icon(class_id):

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


def get_recommendation(class_id):

    recommendations = {

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
        11: "Follow the right-of-way rules at the intersection.",
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

    return recommendations.get(
        class_id,
        "Follow the road signs and drive carefully."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Logo

    st.markdown("""
    <div style="
        text-align:center;
        padding:5px 0 20px;
    ">

        <div style="
            width:72px;
            height:72px;
            margin:auto;
            border-radius:20px;

            display:flex;
            align-items:center;
            justify-content:center;

            background:
                linear-gradient(
                    135deg,
                    #0891b2,
                    #2563eb
                );

            font-size:38px;

            box-shadow:
                0 10px 30px
                rgba(34,211,238,0.20);
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
            font-size:0.72rem;
            margin:0;
            letter-spacing:1px;
            font-weight:600;
        ">
            CNN CLASSIFICATION SYSTEM
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="
        color:#64748b;
        font-size:0.70rem;
        font-weight:700;
        letter-spacing:1px;
        margin-bottom:8px;
    ">
        NAVIGATION
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Predict Sign",
            "📊 Model Performance",
            "🚦 Sign Classes",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if model_loaded:

        st.markdown("""
        <div class="status-online">
            ● MODEL ONLINE
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="status-warning">
            ● MODEL ERROR
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:rgba(15,23,42,0.70);
        border:1px solid rgba(148,163,184,0.12);
        border-radius:15px;
        padding:15px;

        color:#94a3b8;

        font-size:0.76rem;
        line-height:1.65;
    ">

        <div style="
            color:#67e8f9;
            font-weight:700;
            margin-bottom:9px;
        ">
            ⚙️ SYSTEM INFO
        </div>

        <b style="color:#cbd5e1;">
            Technology
        </b>
        <br>
        TensorFlow • Keras • CNN

        <br><br>

        <b style="color:#cbd5e1;">
            Input
        </b>
        <br>
        32 × 32 RGB Image

        <br><br>

        <b style="color:#cbd5e1;">
            Classes
        </b>
        <br>
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

        <div style="
            display:flex;
            align-items:center;
            gap:40px;
        ">

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

            <div class="hero-traffic">
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

    cards = [
        ("Traffic Classes", "43", "Sign Categories"),
        ("Training Images", "39,209", "Original Dataset"),
        ("Test Images", "12,630", "Evaluation Dataset"),
        ("Test Accuracy", "96.44%", "CNN Performance")
    ]

    for col, data in zip(
        [c1, c2, c3, c4],
        cards
    ):

        with col:

            st.markdown(f"""
            <div class="kpi-card">

                <div class="kpi-title">
                    {data[0]}
                </div>

                <div class="kpi-value">
                    {data[1]}
                </div>

                <div class="kpi-sub">
                    {data[2]}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">CNN Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Image processing pipeline used by the deep learning model</div>',
        unsafe_allow_html=True
    )

    a1, a2, a3, a4, a5 = st.columns(5)

    architecture = [
        ("🖼️", "Input", "32 × 32 × 3"),
        ("🧩", "Conv2D", "32 Filters"),
        ("🧠", "Conv2D", "64 + 128 Filters"),
        ("🔄", "Dense", "128 Neurons"),
        ("🚦", "Output", "43 Classes")
    ]

    for col, data in zip(
        [a1, a2, a3, a4, a5],
        architecture
    ):

        with col:

            st.markdown(f"""
            <div class="arch-step">

                <div class="arch-icon">
                    {data[0]}
                </div>

                <div class="arch-title">
                    {data[1]}
                </div>

                <div class="arch-text">
                    {data[2]}
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                🎯 Project Objective
            </div>

            <p style="
                color:#cbd5e1;
                line-height:1.8;
            ">
                The objective of this project is to automatically
                recognize and classify road traffic signs from images
                using a Convolutional Neural Network.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">
                The CNN learns visual patterns such as shape, color,
                symbols and road-sign structures from labeled images.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="card">

            <div class="section-title">
                ⚡ System Status
            </div>

            <br>

            <span class="status-online">
                ● MODEL READY
            </span>

            <p style="
                color:#94a3b8;
                font-size:0.84rem;
                line-height:1.6;
                margin-top:14px;
            ">
                The trained CNN model is ready to classify
                uploaded traffic-sign images.
            </p>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PREDICT SIGN
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
            Upload a traffic-sign image and the trained CNN model
            will analyze it and return the predicted category,
            confidence score and top alternative predictions.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:

        st.error(
            "Model could not be loaded. Please check the model file."
        )

    else:

        left, right = st.columns(
            [1, 1],
            gap="large"
        )

        with left:

            st.markdown(
                '<div class="section-title">📤 Upload Image</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">JPG, JPEG and PNG images are supported</div>',
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Upload Traffic Sign",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                label_visibility="collapsed"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.50,
                max_value=0.99,
                value=0.70,
                step=0.01
            )

            st.caption(
                "Predictions below this confidence level are marked "
                "as uncertain."
            )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            with left:

                st.image(
                    image,
                    caption="Uploaded Traffic Sign",
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
                        {get_sign_icon(best["class_id"])}
                    </div>

                    <div class="prediction-label">
                        Predicted Class
                    </div>

                    <div class="prediction-name">
                        {best["name"]}
                    </div>

                    <div class="confidence">
                        {best["confidence"] * 100:.2f}%
                    </div>

                    <div style="
                        color:#94a3b8;
                        margin-top:4px;
                    ">
                        Confidence Score
                    </div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if best["confidence"] >= confidence_threshold:

                    st.success(
                        "✓ High-confidence prediction"
                    )

                else:

                    st.warning(
                        "⚠ Low-confidence prediction. "
                        "The image may not match the learned classes."
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
                        {get_recommendation(best["class_id"])}
                    </p>

                </div>
                """, unsafe_allow_html=True)

            # ====================================================
            # TOP 3
            # ====================================================

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">📊 Top 3 Predictions</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">Most probable classes returned by the CNN</div>',
                unsafe_allow_html=True
            )

            for rank, result in enumerate(
                results,
                start=1
            ):

                col1, col2, col3 = st.columns(
                    [0.10, 0.68, 0.22]
                )

                with col1:

                    st.markdown(
                        f"""
                        <div style="
                            color:#67e8f9;
                            font-size:1.3rem;
                            font-weight:800;
                            margin-top:10px;
                        ">
                            #{rank}
                        </div>
                        """,
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
                            {get_sign_icon(result["class_id"])}
                            {result["name"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        result["confidence"]
                    )

                with col3:

                    st.markdown(
                        f"""
                        <div style="
                            text-align:right;
                            color:#67e8f9;
                            font-weight:800;
                            font-size:1rem;
                            margin-top:8px;
                        ">
                            {result["confidence"] * 100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:

            st.markdown("""
            <div class="card" style="
                text-align:center;
                padding:55px 30px;
            ">

                <div style="
                    font-size:4.5rem;
                    margin-bottom:15px;
                ">
                    📷
                </div>

                <h2 style="
                    color:white;
                    margin-bottom:10px;
                ">
                    Upload an Image to Begin
                </h2>

                <p style="
                    color:#94a3b8;
                    max-width:600px;
                    margin:auto;
                    line-height:1.7;
                ">
                    Select a traffic-sign image from your computer.
                    The trained CNN will analyze the image and
                    display the predicted traffic-sign category.
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
            Evaluation results obtained from the independent test
            dataset containing 12,630 unseen traffic-sign images.
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

    for col, metric in zip(
        [p1, p2, p3, p4],
        metrics
    ):

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
                    <td style="padding:10px 0;">
                        Architecture
                    </td>

                    <td style="text-align:right;">
                        CNN
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">
                        Input Shape
                    </td>

                    <td style="text-align:right;">
                        32 × 32 × 3
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">
                        Output Classes
                    </td>

                    <td style="text-align:right;">
                        43
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">
                        Optimizer
                    </td>

                    <td style="text-align:right;">
                        Adam
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">
                        Loss Function
                    </td>

                    <td style="text-align:right;">
                        Sparse Categorical Crossentropy
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">
                        Epochs
                    </td>

                    <td style="text-align:right;">
                        15
                    </td>
                </tr>

                <tr>
                    <td style="padding:10px 0;">
                        Batch Size
                    </td>

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

            <p style="
                color:#94a3b8;
                line-height:1.8;
                margin-top:18px;
            ">
                The CNN achieved
                <b style="color:#67e8f9;">
                    96.44% test accuracy
                </b>
                on 12,630 unseen images.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">
                The model correctly classified
                <b style="color:white;">
                    12,180 images
                </b>
                while
                <b style="color:white;">
                    450 images
                </b>
                were incorrectly classified.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">
                The macro F1-score was
                <b style="color:#67e8f9;">
                    94.41%
                </b>,
                showing strong classification performance
                across the 43 classes.
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
            margin-top:22px;
            flex-wrap:wrap;
        ">

            <div class="arch-step" style="flex:1;">
                <div class="arch-icon">🖼️</div>
                <div class="arch-title">Input</div>
                <div class="arch-text">RGB Image</div>
            </div>

            <div style="
                color:#22d3ee;
                font-size:25px;
            ">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                <div class="arch-icon">🧩</div>
                <div class="arch-title">Convolution</div>
                <div class="arch-text">Feature Extraction</div>
            </div>

            <div style="
                color:#22d3ee;
                font-size:25px;
            ">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                <div class="arch-icon">🔄</div>
                <div class="arch-title">Pooling</div>
                <div class="arch-text">Downsampling</div>
            </div>

            <div style="
                color:#22d3ee;
                font-size:25px;
            ">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                <div class="arch-icon">🧠</div>
                <div class="arch-title">Dense</div>
                <div class="arch-text">Classification</div>
            </div>

            <div style="
                color:#22d3ee;
                font-size:25px;
            ">
                →
            </div>

            <div class="arch-step" style="flex:1;">
                <div class="arch-icon">🚦</div>
                <div class="arch-title">Output</div>
                <div class="arch-text">43 Classes</div>
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
            Explore all traffic-sign categories recognized by
            the trained CNN model.
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

            filtered.append(
                (class_id, name)
            )

    if not filtered:

        st.warning(
            "No matching traffic-sign class found."
        )

    else:

        for start in range(
            0,
            len(filtered),
            3
        ):

            row = filtered[
                start:start + 3
            ]

            cols = st.columns(3)

            for col, item in zip(
                cols,
                row
            ):

                class_id, name = item

                with col:

                    st.markdown(f"""
                    <div class="card">

                        <div style="
                            font-size:2rem;
                        ">
                            {get_sign_icon(class_id)}
                        </div>

                        <div style="
                            color:#67e8f9;
                            font-size:0.72rem;
                            font-weight:700;
                            margin-top:8px;
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
            A computer vision project demonstrating the practical
            application of Convolutional Neural Networks for
            automated traffic-sign classification.
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
                Traffic signs provide important information to
                drivers and intelligent transportation systems.
                Automatic recognition can help computer vision
                systems understand road environments.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">
                This project uses deep learning to recognize
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
                A CNN model is trained using labeled traffic-sign
                images. During prediction, the uploaded image is
                resized, normalized and passed through the network.
            </p>

            <p style="
                color:#94a3b8;
                line-height:1.8;
            ">
                The final softmax layer produces probabilities
                for all 43 learned traffic-sign categories.
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

            <span class="status-online">
                Python
            </span>

            <span class="status-online">
                TensorFlow
            </span>

            <span class="status-online">
                Keras
            </span>

            <span class="status-online">
                CNN
            </span>

            <span class="status-online">
                NumPy
            </span>

            <span class="status-online">
                Pillow
            </span>

            <span class="status-online">
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

        <div style="
            margin-top:18px;
            line-height:2;
        ">

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

    Final Year Project • Academic Showcase

</div>
""", unsafe_allow_html=True)
