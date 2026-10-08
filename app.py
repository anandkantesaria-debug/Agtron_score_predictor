import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
st.write("Testing")
st.set_page_config(
    page_title="Coffee Color Predictor",
    page_icon="☕",
    layout="wide"
)
# LOAD MODELS
try:
    agtron_model = CatBoostRegressor()
    agtron_model.load_model("models/agtron_model.cbm")
except Exception as e:
    st.error(f"Critical Error: Failed to load machine learning models. Details: {e}")
    st.stop()
# CUSTOM CSS
st.markdown("""
<style>

.big-font {
    font-size:32px !important;
    font-weight:bold;
}

.stButton>button {
    width:100%;
    height:80px;
    font-size:32px;
    font-weight:bold;
    background-color:#2E8B57;
    color:white;
    border-radius:10px;
}

.result-box {
    padding:20px;
    border-radius:10px;
    background-color:#F0F2F6;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("☕ Coffee Roast Predictor")

st.markdown(
    "Enter roast settings and press **PREDICT**."
)
# INPUTS
col1, col2 = st.columns(2)

with col1:

    month = st.selectbox(
        "Month",
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12"
        ]
    )

    blend = st.text_input(
        "Blend",
        value="284"
    )

    consumed = st.number_input(
        "Consumed Weight (lb)",
        min_value=0.0,
        value=500.0,
        step=1.0
    )

with col2:

    temp = st.number_input(
        "Roast Temperature (°F)",
        min_value=300.0,
        value=440.0,
        step=1.0
    )

    quench = st.number_input(
        "Quench Applied (lb)",
        min_value=0.0,
        value=73.0,
        step=1.0
    )

predict_button = st.button(
    "PREDICT"
)


# PREDICTION

if predict_button:
    try:
        cleaned_blend = str(blend).strip()
        blend = str(int(float(cleaned_blend)))
    except (ValueError, TypeError):
        st.error("Invalid input format detected for 'Blend'. Please enter a valid number.")
        st.stop()


    #st.write("Selected Blend:", blend)

    
    input_data = pd.DataFrame({
        "Consumed": [consumed],
        "Temp": [temp],
        "Quench": [quench],
        "Month": [str(month)],
        "Blend": [blend]        
    })

    try:
        agtron_prediction = float(agtron_model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Prediction calculation failed: {e}")
        st.stop()


    st.divider()

    res1 = st.columns(1)

    st.metric(
        "Predicted Agtron",
        f"{agtron_prediction:.1f}"
    )