import streamlit as st
import mlflow.catboost
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import mlflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

st.set_page_config(
    page_title="Coffee Color Predictor",
    page_icon="☕",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    return model.load_model("models/agtron_model.cbm")
# mlflow.catboost.load_model("models:/AgtronModel@champion")
# LOAD MODELS
try:
    agtron_model = load_model()
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

st.title("☕ Coffee Color Predictor")

tab1, tab2 = st.tabs([
    "Prediction",
    "Recommendation"
])

with tab1:

    st.header("Agtron Prediction")
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
with tab2:

    st.header("Roast Recommendation")

    target_agtron = st.number_input(
        "Target Agtron",
        min_value=15.0,
        max_value=65.0,
        value=45.0,
        step=0.1
    )

    st.write(f"Target Agtron: {target_agtron}")

    recommend_button = st.button(
        "Recommend Settings"
    )
    if recommend_button:

        input_data = pd.DataFrame({
                    "Consumed": [consumed],
                    "Temp": [temp],
                    "Quench": [quench],
                    "Month": [str(month)],
                    "Blend": [blend]        
                })


        current_prediction = float(agtron_model.predict(input_data)[0])

        st.subheader("Current Roast")

        st.write(
            f"Current Predicted Agtron: {current_prediction:.1f}"
        )
        difference = target_agtron - current_prediction
        st.write(
            f"Gap to Target: {difference:.1f}"
        )

        best_temp = temp
        best_quench = quench

        best_prediction = current_prediction

        best_error = abs(target_agtron - current_prediction)
        recommendations = []

        
        # recommended_temp = temp
        # recommended_quench = quench

        for temp_change in range(-11, 10):

            for quench_change in range(-6, 5):

                candidate_df = input_data.copy()

                candidate_df["Temp"] = temp + temp_change
                candidate_df["Quench"] = quench + quench_change

                candidate_prediction = agtron_model.predict(candidate_df)[0]

                candidate_error = abs(target_agtron - candidate_prediction)
                recommendations.append({
                        "Temp": temp + temp_change,
                        "Quench": quench + quench_change,
                        "Expected Agtron": round(candidate_prediction, 1),
                        "Error": round(candidate_error, 1)
                })

                if candidate_error < best_error:

                    best_error = candidate_error

                    best_prediction = candidate_prediction


                    best_temp = temp + temp_change
                    best_quench = quench + quench_change

        # if difference > 0:
        #     recommended_temp += 2
        #     recommended_quench += 5

        # elif difference < 0:
        #     recommended_temp -= 2
        #     recommended_quench -= 5
        recommendation_df = pd.DataFrame(recommendations)
        recommendation_df = recommendation_df.sort_values("Error")
        st.subheader("Optimized Recommendation")

        st.write(f"Recommended Temp: {best_temp}")

        st.write(f"Recommended Quench: {best_quench}")

        st.success(f"Expected Agtron: {best_prediction:.1f}")

        st.write(f"Distance From Target: {best_error:.1f}")

        st.subheader("Top 5 Recommendations")

        st.dataframe(
            recommendation_df.head(5),
            use_container_width=True
        )
        
        # st.subheader("Recommended Changes")

        # st.write(f"Recommended Temp: {recommended_temp}")

        # st.write(
        #     f"Recommended Time: {recommended_quench}"
        # )

        # recommended_df = input_data.copy()
        # recommended_df["Temp"] = recommended_temp
        # recommended_df["Quench"] = recommended_quench

        # recommended_prediction = agtron_model.predict(recommended_df)[0]

        # st.subheader("Expected Result")

        # st.success(f"Expected Agtron: {recommended_prediction:.1f}")