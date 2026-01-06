import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv()

model = pickle.load(open("churn_model.pkl", "rb"))
explainer = pickle.load(open("shap_explainer.pkl", "rb"))
columns = pickle.load(open("encoder_columns.pkl", "rb"))
metrics = pickle.load(open("model_metrics.pkl", "rb"))

st.set_page_config(page_title="AI Customer Retention", layout="wide")

st.markdown("""
<style>
body { background-color: #f6f7fb; }
.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.badge-risk {
    background:#fee2e2;
    color:#991b1b;
    padding:6px 14px;
    border-radius:20px;
}
.badge-safe {
    background:#dcfce7;
    color:#166534;
    padding:6px 14px;
    border-radius:20px;
}
.email-box {
    background:#f9fafb;
    padding:16px;
    border-radius:10px;
    border-left:5px solid #6366f1;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;">
    <h1>🏢 TELCO AI SOLUTIONS</h1>
    <p>GenAI-Powered Customer Retention Platform</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📥 Customer Input")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 3)
monthly = st.sidebar.slider("Monthly Charges", 20, 150, 120)
total = st.sidebar.slider("Total Charges", 20, 10000, 300)
tech = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "TechSupport": 1 if tech == "Yes" else 0,
    "Contract": ["Month-to-month", "One year", "Two year"].index(contract)
}

input_df = pd.DataFrame([input_data])

for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

churn_prob = float(model.predict_proba(input_df)[0][1])
prediction = "CHURN" if churn_prob > 0.5 else "SAFE"
email_confidence = round((1 - churn_prob) * 100, 1)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Churn Risk Analysis")
    st.progress(int(churn_prob * 100))
    st.metric("Churn Probability", f"{churn_prob:.2f}")

    if prediction == "CHURN":
        st.markdown("<span class='badge-risk'>HIGH RISK</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-safe'>LOW RISK</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

reasons = []

with col2:
    if prediction == "CHURN":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Why Customer May Leave")

        if tenure < 6:
            reasons.append("Low customer tenure")
            st.write("• Low customer tenure")

        if monthly > 100:
            reasons.append("High monthly charges")
            st.write("• High monthly charges")

        if tech == "No":
            reasons.append("No technical support")
            st.write("• No technical support")

        if contract == "Month-to-month":
            reasons.append("No long-term contract")
            st.write("• Month-to-month contract")

        if not reasons:
            st.write("• Usage pattern indicates churn risk")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
c2.metric("Precision", f"{metrics['precision']:.2f}")
c3.metric("Recall", f"{metrics['recall']:.2f}")
c4.metric("F1 Score", f"{metrics['f1']:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("✉️ AI Retention Email")

if st.button("Generate AI Email"):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("GenAI not connected. Showing fallback email.")
        st.markdown(
            "<div class='email-box'>"
            "Hi! We truly value your association with us. "
            "We noticed some concerns in your recent experience and would love to "
            "offer you a special discount along with priority support to serve you better."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        try:
            client = OpenAI(api_key=api_key)

            prompt = f"""
            You are a customer success manager.
            A customer with tenure {tenure} months is at risk of leaving.
            Reasons: {', '.join(reasons)}.
            Write a polite, empathetic retention email under 100 words.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )

            st.markdown(
                f"<div class='email-box'>{response.choices[0].message.content}</div>",
                unsafe_allow_html=True
            )

        except Exception:
            st.warning("⚠️ GenAI quota exceeded. Showing fallback email.")
            st.markdown(
                "<div class='email-box'>"
                "Hi! We value your loyalty. To ensure a better experience, "
                "we’d like to offer you a personalized discount and dedicated support."
                "</div>",
                unsafe_allow_html=True
            )

    st.caption(f"🧠 AI Confidence Score: {email_confidence}%")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr>
<p style="text-align:center; font-size:13px; color:gray;">
© 2026 Telco AI Solutions | ML • Explainable AI • GenAI
</p>
""", unsafe_allow_html=True)


