import streamlit as st
import numpy as np
import re
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ تحميل الموديل والتوكنيزر
model = load_model("sentiment_cnn_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# ✅ دالة تنظيف النص
def clean_review(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # إزالة الرموز والعلامات
    return text

# ✅ إعداد واجهة Streamlit
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")
st.markdown("<h1 style='text-align: center; color:#4CAF50;'>🔍 Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>أدخل جملة لتحليل المشاعر ومعرفة إذا كانت إيجابية أم سلبية أم محايدة</p>", unsafe_allow_html=True)

# ✅ إدخال المستخدم
user_input = st.text_area("📝 اكتب الجملة هنا:", height=100)

# ✅ زر التنبؤ
if st.button("🔍 تحليل المشاعر"):
    if not user_input.strip():
        st.warning("من فضلك أدخل جملة.")
    else:
        cleaned = clean_review(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)

        prediction = model.predict(padded)[0]

        # لو الموديل بيصنف 3 مشاعر: Negative, Neutral, Positive
        if len(prediction) == 3:
            labels = ['Negative', 'Neutral', 'Positive']
            predicted_class = labels[np.argmax(prediction)]
        else:
            predicted_class = "Positive" if prediction >= 0.5 else "Negative"

        # ✅ عرض النتيجة
        st.success(f"🎯 التوقع: **{predicted_class}**")
