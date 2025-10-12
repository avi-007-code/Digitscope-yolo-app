import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

# === Initialize theme session ===
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "🌙 Dark Mode"

# === Sidebar Theme Switcher ===
theme = st.sidebar.radio("🖌️ Choose Theme", ["🌙 Dark Mode", "☀️ Light Mode"])

# Apply selected theme to session state
if theme != st.session_state.theme_mode:
    st.session_state.theme_mode = theme
    st.rerun()  # Re-render the app after theme change

# === Apply theme CSS ===
def set_custom_css(theme):
    if "Dark" in theme:
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');

        html, body, [class*="css"] {
            font-family: 'Fira Code', monospace;
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #f5f5f5;
        }

        .stButton>button {
            background-color: #00adb5;
            color: white;
        }

        .stButton>button:hover {
            background-color: #393e46;
            color: #eeeeee;
        }

        .stImage > img {
            border: 2px solid #00adb5;
            border-radius: 10px;
        }
        </style>
        """
    else:
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');

        html, body, [class*="css"] {
            font-family: 'Fira Code', monospace;
            background-color: #fdfdfd;
            color: #222;
        }

        .stButton>button {
            background-color: #007acc;
            color: white;
        }

        .stButton>button:hover {
            background-color: #005a8d;
            color: #ffffff;
        }

        .stImage > img {
            border: 2px solid #007acc;
            border-radius: 10px;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

set_custom_css(st.session_state.theme_mode)

# === Load YOLO model ===
@st.cache_resource
def load_model():
    return YOLO("number_detection.pt")

model = load_model()

# === App Main Content ===
st.title("🔍 DigitScope: Smart Number Detector")
st.markdown("Upload an image and let **DigitScope** reveal the hidden numbers using YOLOv11n AI vision. 🔢⚡")

uploaded_file = st.file_uploader("📤 Drop your number image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    # ✅ Updated: use_container_width replaces use_column_width
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    if st.button("🚀 Run Detection"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            results = model(tmp_file.name)

        result_img = results[0].plot()
        # ✅ Updated here too
        st.image(result_img, caption="📍 Detected Numbers", use_container_width=True)

        st.subheader("🔎 Prediction Summary")
        boxes = results[0].boxes
        names = model.names

        if boxes and boxes.cls.numel() > 0:
            for cls in boxes.cls:
                label = names[int(cls)]
                st.write(f"🔹 Digit Found: `{label}`")
        else:
            st.warning("⚠️ No digits detected. Try a clearer image.")
