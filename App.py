import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load the model safely
# -------------------------
try:
    model_data = joblib.load("diabetes_model.pkl")
    # Handle both dict or direct model
    model = model_data['model'] if isinstance(model_data, dict) else model_data
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# -------------------------
# Page settings
# -------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("🩺 Diabetes Prediction Dashboard")
st.write("A professional tool to assess diabetes risk based on patient metrics.")

# -------------------------
# Sidebar input form
# -------------------------
st.sidebar.header("Patient Information")

with st.sidebar.form(key="patient_form"):
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    submit_button = st.form_submit_button(label="Predict")

# -------------------------
# Prediction Logic
# -------------------------
if submit_button:
    try:
        features = np.array([[glucose, blood_pressure, insulin, bmi, age]])
        prediction = model.predict(features)[0]

        col1, col2 = st.columns([2, 1])

        # Show result
        with col1:
            if prediction == 1:
                st.error("⚠ The patient is likely to have diabetes.")
                st.subheader("Precautions:")
                st.write(
                    """
                    - Maintain a healthy, balanced diet  
                    - Exercise regularly  
                    - Monitor blood sugar frequently  
                    - Avoid sugary foods and drinks  
                    - Consult your doctor regularly  
                    """
                )
            else:
                st.success("✅ The patient is unlikely to have diabetes.")
                st.write("Continue maintaining a healthy lifestyle!")

        # Show graph
        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.bar(["Glucose", "BMI"], [glucose, bmi], color=["#FF9999", "#66B2FF"])
            ax.axhline(140, color="red", linestyle="--", label="High Glucose Risk")
            ax.axhline(25, color="blue", linestyle="--", label="Healthy BMI")
            ax.set_ylabel("Value")
            ax.set_title("Patient Health Metrics")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Made with ❤ using Streamlit")

st.markdown("**Developed by:** Mehak Naz  |  📧 nmehak875@gmail.com")
