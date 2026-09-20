import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Traffic Sign AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.08);
    text-align: center;
    min-height: 120px;
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
}

.metric-label {
    font-size: 15px;
    opacity: 0.7;
}

.prediction-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.3);
    background: rgba(128,128,128,0.08);
    text-align: center;
}

.prediction-name {
    font-size: 30px;
    font-weight: 800;
}

.confidence {
    font-size: 22px;
    font-weight: 700;
}

.section-title {
    font-size: 26px;
    font-weight: 750;
    margin-top: 15px;
}

.small-text {
    opacity: 0.7;
}

</style>
""", unsafe_allow_html=True)


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
# RECOMMENDATIONS
# =========================================================

recommendations = {

    "Speed limit (20 km/h)": "Maintain your speed at or below 20 km/h.",
    "Speed limit (30 km/h)": "Maintain your speed at or below 30 km/h.",
    "Speed limit (50 km/h)": "Maintain your speed at or below 50 km/h.",
    "Speed limit (60 km/h)": "Maintain your speed at or below 60 km/h.",
    "Speed limit (70 km/h)": "Maintain your speed at or below 70 km/h.",
    "Speed limit (80 km/h)": "Maintain your speed at or below 80 km/h.",
    "End of speed limit (80 km/h)": "The previous 80 km/h speed restriction has ended. Follow the next applicable speed sign.",
    "Speed limit (100 km/h)": "Maintain your speed at or below 100 km/h.",
    "Speed limit (120 km/h)": "Maintain your speed at or below 120 km/h.",
    "No passing": "Do not overtake other vehicles in this zone.",
    "No passing for vehicles over 3.5 tons": "Heavy vehicles over 3.5 tons should not overtake.",
    "Right-of-way at intersection": "Give priority according to the intersection rules.",
    "Priority road": "You are travelling on a priority road. Continue carefully.",
    "Yield": "Slow down and give way to other road users.",
    "Stop": "Come to a complete stop and check the road before proceeding.",
    "No vehicles": "Vehicles are not permitted in this area.",
    "Vehicles over 3.5 tons prohibited": "Vehicles over 3.5 tons should not enter this area.",
    "No entry": "Do not enter this road or restricted area.",
    "General caution": "Drive carefully and watch for possible hazards.",
    "Dangerous curve left": "Slow down and prepare for a dangerous left curve.",
    "Dangerous curve right": "Slow down and prepare for a dangerous right curve.",
    "Double curve": "Reduce speed and prepare for consecutive curves.",
    "Bumpy road": "Slow down because the road surface may be uneven.",
    "Slippery road": "Reduce speed and avoid sudden braking or steering.",
    "Road narrows on the right": "Reduce speed and be careful as the road becomes narrower.",
    "Road work": "Slow down and watch for road construction workers and obstacles.",
    "Traffic signals": "Be alert for traffic lights and follow their signals.",
    "Pedestrians": "Watch carefully for pedestrians and reduce speed if necessary.",
    "Children crossing": "Slow down and watch for children crossing the road.",
    "Bicycles crossing": "Watch for cyclists and maintain a safe distance.",
    "Beware of ice/snow": "Drive carefully because the road may be icy or snowy.",
    "Wild animals crossing": "Watch for animals crossing the road.",
    "End of all speed and passing limits": "Previous speed and passing restrictions have ended. Follow the next applicable sign.",
    "Turn right ahead": "Prepare to turn right ahead.",
    "Turn left ahead": "Prepare to turn left ahead.",
    "Ahead only": "Continue straight ahead.",
    "Go straight or right": "You may continue straight or turn right.",
    "Go straight or left": "You may continue straight or turn left.",
    "Keep right": "Keep to the right side of the road.",
    "Keep left": "Keep to the left side of the road.",
    "Roundabout mandatory": "Enter the roundabout according to the indicated direction.",
    "End of no passing": "The previous no-passing restriction has ended.",
    "End of no passing by vehicles over 3.5 tons": "The heavy-vehicle no-passing restriction has ended."
}


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "traffic_sign_cnn_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🚦 Traffic Sign AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "💡 Recommendation",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🎨 Appearance")

dark_mode = st.sidebar.toggle(
    "🌙 Night Mode",
    value=False
)

if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)


st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Model Status")
st.sidebar.success("Model Loaded")

st.sidebar.caption("CNN • 43 Traffic Sign Classes")
st.sidebar.caption("Test Accuracy: 96.44%")


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🚦 Traffic Sign AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Traffic Sign Classification using Convolutional Neural Network</div>',
        unsafe_allow_html=True
    )

    st.markdown("## 📊 Model Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">43</div>
            <div class="metric-label">Traffic Sign Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">96.44%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">12,630</div>
            <div class="metric-label">Test Images</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">CNN</div>
            <div class="metric-label">Deep Learning Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("## 🧠 Model Architecture")

        st.info("""
        **Input Layer**  
        32 × 32 × 3 RGB Image

        ↓

        **Convolutional Layer 1**  
        32 Filters + ReLU

        ↓

        **Max Pooling**

        ↓

        **Convolutional Layer 2**  
        64 Filters + ReLU

        ↓

        **Max Pooling**

        ↓

        **Convolutional Layer 3**  
        128 Filters + ReLU

        ↓

        **Flatten + Dense Layer**

        ↓

        **Output Layer**  
        43 Classes + Softmax
        """)

    with col2:

        st.markdown("## ✨ Key Features")

        st.success("""
        ✅ Image-based traffic sign recognition

        ✅ CNN-based deep learning model

        ✅ 43 different traffic sign classes

        ✅ Real-time image prediction

        ✅ Confidence score

        ✅ Top-3 predictions

        ✅ Driving recommendation

        ✅ Day / Night interface

        ✅ User-friendly Streamlit dashboard
        """)

    st.markdown("---")

    st.markdown("## 🚀 How It Works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 1️⃣ Upload")
        st.caption("Upload a traffic sign image.")

    with step2:
        st.markdown("### 2️⃣ Process")
        st.caption("Image is resized to 32×32 and normalized.")

    with step3:
        st.markdown("### 3️⃣ Predict")
        st.caption("CNN predicts the traffic sign class.")

    with step4:
        st.markdown("### 4️⃣ Recommend")
        st.caption("System provides a driving recommendation.")


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.markdown(
        '<div class="main-title">🔮 Traffic Sign Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Upload an image and let the CNN identify the traffic sign.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📷 Upload Traffic Sign Image",
        type=["jpg", "jpeg", "png"]
    )

    threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1.5])

        with col1:

            st.markdown("### 📷 Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown("### 🤖 AI Analysis")

            img = image.resize((32, 32))

            img_array = np.array(img).astype("float32") / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            prediction = model.predict(
                img_array,
                verbose=0
            )[0]

            class_id = int(np.argmax(prediction))

            confidence = float(
                np.max(prediction)
            )

            top_indices = np.argsort(
                prediction
            )[-3:][::-1]

            if confidence >= threshold:

                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-name">
                            🚦 {class_names[class_id]}
                        </div>
                        <br>
                        <div class="confidence">
                            Confidence: {confidence * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    f"Class ID: {class_id}"
                )

            else:

                st.warning(
                    "⚠️ The model is not confident enough to recognize this image as a traffic sign."
                )

                st.info(
                    f"Maximum model confidence: {confidence * 100:.2f}%"
                )

        st.markdown("---")

        if confidence >= threshold:

            st.markdown("## 🏆 Top 3 Predictions")

            for rank, idx in enumerate(top_indices, start=1):

                probability = float(
                    prediction[idx]
                )

                st.progress(
                    probability,
                    text=f"{rank}. {class_names[idx]} — {probability * 100:.2f}%"
                )

            st.markdown("---")

            st.markdown("## 💡 Driving Recommendation")

            recommendation = recommendations.get(
                class_names[class_id],
                "Follow the applicable traffic rules and drive safely."
            )

            st.info(
                f"🚘 {recommendation}"
            )

            st.caption(
                "⚠️ This recommendation is for educational/demo purposes. "
                "Always follow actual road signs and local traffic regulations."
            )


# =========================================================
# RECOMMENDATION
# =========================================================

elif page == "💡 Recommendation":

    st.markdown(
        '<div class="main-title">💡 Traffic Sign Recommendation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Select a traffic sign to view its driving guidance.</div>',
        unsafe_allow_html=True
    )

    selected_sign = st.selectbox(
        "🚦 Select Traffic Sign",
        class_names
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🚦 Selected Sign")

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-name">
                    {selected_sign}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown("### 💡 Recommendation")

        st.info(
            recommendations.get(
                selected_sign,
                "Follow the applicable traffic rules and drive safely."
            )
        )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Traffic Sign Classification using Deep Learning</div>',
        unsafe_allow_html=True
    )

    st.markdown("## 🎯 Project Objective")

    st.write("""
    The objective of this project is to develop a deep learning-based
    system that can automatically recognize and classify traffic signs
    from images.
    """)

    st.markdown("## 🧠 Technology Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.markdown("### 🐍 Python")
        st.caption("Programming Language")

    with tech2:
        st.markdown("### 🧠 TensorFlow")
        st.caption("Deep Learning Framework")

    with tech3:
        st.markdown("### 🔥 CNN")
        st.caption("Classification Model")

    with tech4:
        st.markdown("### 🎈 Streamlit")
        st.caption("Web Application")

    st.markdown("---")

    st.markdown("## 📊 Dataset")

    st.write("""
    The model was trained using a German Traffic Sign Recognition
    dataset containing 43 different traffic sign classes.
    """)

    st.markdown("## 📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Test Accuracy",
            "96.44%"
        )

    with col2:
        st.metric(
            "Correct Predictions",
            "12,180"
        )

    with col3:
        st.metric(
            "Wrong Predictions",
            "450"
        )

    st.markdown("---")

    st.markdown("## 👨‍💻 Project Information")

    st.write("""
    **Project:** Traffic Sign Classification Using Deep Learning

    **Model:** Custom Convolutional Neural Network (CNN)

    **Input Size:** 32 × 32 RGB

    **Number of Classes:** 43

    **Deployment:** Streamlit Community Cloud
    """)

    st.info(
        "This application is developed as an academic/project demonstration "
        "of image classification using deep learning."
    )
