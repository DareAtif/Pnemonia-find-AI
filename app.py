import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Pneumonia-AI",
    page_icon="",
    layout="centered"
)

# -----------------------------
# BACKGROUND IMAGE - Using Gradient
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.block-container {
    padding-top: 1rem;
}

/* ========== SIDEBAR - LEMON YELLOW ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff9c4 0%, #fff59d 50%, #fff176 100%) !important;
    border-right: 3px solid #fdd835 !important;
    box-shadow: 4px 0 20px rgba(253, 216, 53, 0.3) !important;
}

[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    color: #1a1a2e !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #1a1a2e !important;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #1a1a2e !important;
}

[data-testid="stSidebar"] .stAlert {
    background: rgba(0,0,0,0.05) !important;
    color: #1a1a2e !important;
}

[data-testid="stSidebar"] .stAlert p {
    color: #1a1a2e !important;
}

/* Sidebar Dividers */
[data-testid="stSidebar"] hr {
    border-color: #fdd835 !important;
    border-width: 2px !important;
}

/* Sidebar Markdown Containers */
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: #1a1a2e !important;
}

/* Sidebar - Warning/Disclaimer Box */
[data-testid="stSidebar"] .stWarning {
    background: rgba(255, 193, 7, 0.3) !important;
    border-left: 4px solid #f9a825 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] .stWarning p {
    color: #1a1a2e !important;
}

/* Sidebar - Info Box */
[data-testid="stSidebar"] .stInfo {
    background: rgba(255, 224, 130, 0.3) !important;
    border-left: 4px solid #fdd835 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] .stInfo p {
    color: #1a1a2e !important;
}

/* Sidebar - Button */
[data-testid="stSidebar"] div.stButton > button {
    background: linear-gradient(135deg, #fdd835, #f9a825) !important;
    color: #1a1a2e !important;
    border: none !important;
    font-weight: bold !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: linear-gradient(135deg, #f9a825, #f57f17) !important;
    transform: scale(1.02);
    transition: all 0.3s ease;
}

/* ========== MAIN CONTENT ========== */
.hero-box {
    background: rgba(15,23,42,0.85);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 25px;
    padding: 25px;
    text-align: center;
    margin-bottom: 25px;
}

.hero-box h1 {
    color: white;
    font-size: 55px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero-box p {
    color: #cbd5e1;
    font-size: 18px;
}

.upload-box {
    background: rgba(15,23,42,0.75);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 20px;
}

.result-card {
    background: rgba(15,23,42,0.85);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(59,130,246,0.4);
}

.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.1);
}

/* Upload button */
div.stFileUploader > div > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

div.stFileUploader > div > button:hover {
    transform: scale(1.02);
    transition: all 0.3s ease;
}

/* Analyze button */
div.stButton > button {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 30px;
    font-weight: bold;
    font-size: 16px;
    width: 100%;
}

div.stButton > button:hover {
    transform: scale(1.02);
    transition: all 0.3s ease;
    box-shadow: 0 10px 20px rgba(245, 87, 108, 0.3);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #1a1a2e;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

/* ========== RESULT COLORS ========== */

/* Result Card - Main Title */
.result-card h2 {
    color: white !important;
    font-weight: bold !important;
    text-shadow: 0 0 30px rgba(0,0,0,0.8) !important;
}

/* Result Card - All Text */
.result-card div {
    color: #e2e8f0 !important;
}

/* Warning Box */
.stWarning {
    background: rgba(220, 38, 38, 0.25) !important;
    border-left: 5px solid #ef4444 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
    padding: 15px !important;
}

.stWarning p, 
.stWarning strong, 
.stWarning b, 
.stWarning span,
.stWarning div {
    color: #ffffff !important;
    font-weight: 600 !important;
}

.stWarning strong {
    color: #fca5a5 !important;
    font-size: 18px !important;
}

/* Success Box */
.stSuccess {
    background: rgba(34, 197, 94, 0.2) !important;
    border-left: 5px solid #22c55e !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
    padding: 15px !important;
}

.stSuccess p, 
.stSuccess strong, 
.stSuccess b, 
.stSuccess span,
.stSuccess div {
    color: #ffffff !important;
    font-weight: 600 !important;
}

.stSuccess strong {
    color: #86efac !important;
    font-size: 18px !important;
}

/* Info Box */
.stInfo {
    background: rgba(59, 130, 246, 0.2) !important;
    border-left: 5px solid #3b82f6 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
    padding: 15px !important;
}

.stInfo p, 
.stInfo strong, 
.stInfo b, 
.stInfo span,
.stInfo div {
    color: #ffffff !important;
    font-weight: 500 !important;
}

.stInfo strong {
    color: #93c5fd !important;
}

/* Metric Container */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    color: #86efac !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
try:
    logo = Image.open("assets/logo.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=200)
except:
    pass

st.markdown("""
<div class="hero-box">
    <h1> PNEUMONIA AI</h1>
    <p>
        Deep Learning Powered Chest X-Ray Analysis<br>
        <span style="font-size: 14px; color: #94a3b8;">Powered by VGG16 Transfer Learning</span>
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    """Load the trained model from local file"""
    local_model_path = "./best_vgg_finetune.keras"
    
    if os.path.exists(local_model_path):
        try:
            model = tf.keras.models.load_model(local_model_path)
            return model
        except Exception as e:
            st.error(f"❌ Could not load model: {e}")
            st.stop()
    else:
        st.error("❌ Model file not found! Please ensure 'best_vgg_finetune.keras' is in the app directory.")
        st.stop()

with st.spinner("🔄 Loading AI Model..."):
    model = load_model()
    st.success("✅ Model loaded successfully!")

# -----------------------------
# SIDEBAR - LEMON YELLOW THEME
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="color: #1a1a2e; font-weight: 800; margin-bottom: 5px;">🫁 Pneumonia AI</h2>
        <p style="color: #1a1a2e; font-size: 14px; opacity: 0.8;">AI-Powered X-Ray Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 About This App")
    st.markdown("""
    **Pneumonia AI** uses deep learning to analyze chest X-ray images.
    
    - 🏥 **Model**: VGG16 (Fine-tuned)
    - 📈 **Accuracy**: 90%+
    - ⚡ **Inference Time**: < 2 seconds
    """)
    
    st.markdown("---")
    
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. 📤 Upload a chest X-ray image
    2. 🔍 Click "Analyze X-Ray"
    3. 📊 Get instant AI results
    """)
    
    st.markdown("---")
    
    # Disclaimer with lemon yellow styling
    st.markdown("""
    <div style="background: rgba(255, 193, 7, 0.2); padding: 15px; border-radius: 10px; border-left: 4px solid #f9a825;">
        <p style="color: #1a1a2e; font-weight: 600; margin: 0;">
            ⚠️ <strong>Disclaimer</strong>
        </p>
        <p style="color: #1a1a2e; font-size: 13px; margin: 5px 0 0 0;">
            This is a <strong>screening tool</strong> only. 
            Always consult a healthcare professional.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Developer")
    st.markdown("""
    <div style="background: rgba(255, 224, 130, 0.3); padding: 10px; border-radius: 8px;">
        <p style="color: #1a1a2e; font-weight: 600; margin: 0;">
            Sayed Atif Hosen
        </p>
        <p style="color: #1a1a2e; font-size: 12px; margin: 3px 0 0 0; opacity: 0.7;">
            AI Engineer | Deep Learning Enthusiast
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Version info
    st.markdown("""
    <div style="text-align: center; padding: 5px;">
        <p style="color: #1a1a2e; font-size: 11px; opacity: 0.6;">
            v1.0 • Powered by TensorFlow
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# MAIN CONTENT
# -----------------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PROCESS IMAGE
# -----------------------------
if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            image,
            caption="📸 Uploaded Chest X-Ray",
            use_container_width=True
        )
    
    if st.button("🔍 Analyze X-Ray", use_container_width=True):
        
        with st.spinner("🧠 Analyzing X-Ray with AI..."):
            
            img = image.resize((224, 224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            
            try:
                prediction = model.predict(img)
                confidence = float(prediction[0][0])
                
                if confidence > 0.5:
                    result = "PNEUMONIA"
                    score = confidence * 100
                    color = "#ef4444"
                    icon = "⚠️"
                else:
                    result = "NORMAL"
                    score = (1 - confidence) * 100
                    color = "#22c55e"
                    icon = "✅"
                
                # Result Card
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Main Result
                st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <h2 style="color: {color} !important; font-size: 42px; font-weight: 800; margin-bottom: 10px; text-shadow: 0 0 30px rgba(0,0,0,0.9);">
                        {icon} {result}
                    </h2>
                    <p style="color: #cbd5e1 !important; font-size: 16px; font-weight: 500;">
                        AI Analysis Complete
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence Score
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        label="Confidence Score",
                        value=f"{score:.2f}%",
                        delta="High confidence" if score > 80 else "Moderate confidence"
                    )
                
                # Result Messages
                if result == "PNEUMONIA":
                    st.warning("""
                    ⚠️ **Signs of pneumonia detected**  
                    Please consult a healthcare professional immediately.
                    """)
                    
                    st.info("""
                    📊 **Analysis Details:**
                    - **AI Confidence:** {:.1f}%
                    - **Model:** VGG16 Fine-tuned
                    - **Recommendation:** Medical consultation required
                    """.format(score))
                    
                else:
                    st.success("""
                    ✅ **No signs of pneumonia detected**  
                    The chest X-ray appears normal.
                    """)
                    
                    st.info("""
                    📊 **Analysis Details:**
                    - **AI Confidence:** {:.1f}%
                    - **Model:** VGG16 Fine-tuned
                    - **Status:** Normal
                    """.format(score))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("Please try again with a different image.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    <p> Pneumonia AI v1.0 | Developed by Sayed Atif Hosen</p>
    <p style="font-size: 12px; opacity: 0.7;">
        This is an AI screening tool. Always consult healthcare professionals.
    </p>
</div>
""", unsafe_allow_html=True)