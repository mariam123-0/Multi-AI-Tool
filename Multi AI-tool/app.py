import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# -------- NLTK Stopwords Setup --------
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
arabic_stopwords = stopwords.words("arabic")

# -------- Streamlit Config --------
st.set_page_config(
    page_title="Multi-AI Tool",
    page_icon="🖥️",
    layout="wide")

# -------- Sidebar --------
st.sidebar.title("🔍 Choose App")
app_choice = st.sidebar.selectbox(
    "Select an application:",
    [
        "📩 Spam Detection",
        "🖼️ Image Feature Selector",
        "📝 Text Summarization in English",
        "📝 تلخيص النص باللغة العربية"])

# -------- Dynamic Theme --------
def apply_theme(app: str) -> str:
    themes = {
        "📩 Spam Detection": """
            <style>
                .stApp { background-color: #1a1a2e; }
                section[data-testid="stSidebar"] { background-color: #16213e; }
                section[data-testid="stSidebar"] * { color: white; }
            </style>
        """,
        "🖼️ Image Feature Selector": """
            <style>
                .stApp { background-color: #2e003e; }
                section[data-testid="stSidebar"] { background-color: #3f0071; }
                section[data-testid="stSidebar"] * { color: #ffd700; }
            </style>
        """,
        "📝 Text Summarization App in English": """
            <style>
                .stApp { background-color: #1c1f26; }
                section[data-testid="stSidebar"] { background-color: #2a2e38; }
                section[data-testid="stSidebar"] * { color: #00bfff; }
            </style>
        """,
        "📝 تطبيق تلخيص النص باللغة العربية": """
            <style>
                .stApp { background-color: #1e1b2f; }
                section[data-testid="stSidebar"] { background-color: #2c2740; }
                section[data-testid="stSidebar"] * { color: #ffcc00; }
            </style>
        """,
        "default": """
            <style>
                .stApp { background-color: #0f2027; }
                section[data-testid="stSidebar"] { background-color: #203a43; }
                section[data-testid="stSidebar"] * { color: #00ffcc; }
            </style>
        """}
    return themes.get(app, themes["default"])
st.markdown(apply_theme(app_choice), unsafe_allow_html=True)

# -------- Spam Detection --------
if app_choice == "📩 Spam Detection":
    st.title("📩 Spam Detection")
    spam_words = ["free", "win", "cash", "offer", "buy now", "click", "credit"]
    def predict_spam(text: str) -> int:
        """Rule-based spam detector"""
        text = text.lower()
        return 0 if any(word in text for word in spam_words) else 1
    user_input = st.text_area("Enter a message:")
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            prediction = predict_spam(user_input)
            
            if prediction == 1:
                st.success("NOT A SPAM! ✅")
            else:
                st.error("SPAM! 💀")


# -------- Image Feature Selector --------
elif app_choice == "🖼️ Image Feature Selector":
    st.title("🖼️ Image Feature Selector")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        feature = st.selectbox(
            "Choose a feature to apply:",
            ["Average Color", "Adjust Brightness", "Flip Image", "Crop Image"])
        if feature == "Average Color":
            img_array = np.array(image)
            avg_color = tuple(map(int, img_array.mean(axis=(0, 1))))
            st.write(f"🎨 Average Color: {avg_color}")
        elif feature == "Adjust Brightness":
            factor = st.slider("Brightness Factor", 0.5, 2.0, 1.0)
            bright_img = ImageEnhance.Brightness(image).enhance(factor)
            st.image(bright_img, caption=f"Brightness Adjusted (factor={factor:.2f})",
            use_container_width=True)
        elif feature == "Flip Image":
            flip_option = st.radio("Flip Direction", ["Horizontal", "Vertical"])
            flipped_img = ImageOps.mirror(image) if flip_option == "Horizontal" else ImageOps.flip(image)
            st.image(flipped_img, caption=f"Flipped {flip_option}", use_container_width=True)
        elif feature == "Crop Image":
            width, height = image.size
            left = st.slider("Left", 0, width // 2, 0)
            top = st.slider("Top", 0, height // 2, 0)
            right = st.slider("Right", width // 2, width, width)
            bottom = st.slider("Bottom", height // 2, height, height)
            cropped_img = image.crop((left, top, right, bottom))
            st.image(cropped_img, caption="Cropped Image", use_container_width=True)

# -------- General Summarizer (English & Arabic) --------
def tfidf_summarize(text: str, k: int, lang: str = "en") -> str:
    """TF-IDF based text summarizer"""
    # Split sentences
    if lang == "ar":
        sentences = re.split(r"[.!؟]", text)
    else:
        sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return "❌ No valid sentences found."
    if k >= len(sentences):
        return text
    if lang == "ar":
        vect = TfidfVectorizer(stop_words=arabic_stopwords)
    else:
        vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    top_idx = np.argsort(scores)[-k:]
    top_sentences = [sentences[i] for i in sorted(top_idx)]
    return ". ".join(top_sentences) + "."

# -------- Text Summarization in English --------
if app_choice == "📝 Text Summarization in English":
    st.title("📝 Text Summarization App in English")
    user_text = st.text_area("Enter text here:")
    num_sentences = st.number_input("Number of sentences in summary:",
                                    min_value=1, max_value=20, value=3)
    if st.button("Summarize"):
        if not user_text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            summary = tfidf_summarize(user_text, num_sentences, lang="en")
            st.subheader("Summary:")
            st.write(summary)

# -------- Text Summarization in Arabic --------
elif app_choice == "📝 تلخيص النص باللغة العربية":
    st.title("📝 تطبيق تلخيص النص باللغة العربية")
    user_text = st.text_area("أدخل النص هنا:")
    num_sentences = st.number_input("عدد الجمل في الملخص:",
                                    min_value=1, max_value=20, value=3)
    if st.button("تلخيص"):
        if not user_text.strip():
            st.warning("يرجى إدخال بعض النص لتلخيصه.")
        else:
            summary = tfidf_summarize(user_text, num_sentences, lang="ar")
            st.subheader("الملخص:")
            st.write(summary)