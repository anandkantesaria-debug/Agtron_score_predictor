import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="Coffee Color Predictor",
    page_icon="☕",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    return model.load_model("models/agtron_model.cbm")


def clean_blend_input(blend_val):
    """Normalize blend input to clean string format (e.g. '284.0' or ' 284 ' -> '284')."""
    try:
        cleaned = str(blend_val).strip()
        return str(int(float(cleaned)))
    except (ValueError, TypeError):
        return None

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
        cleaned_blend = clean_blend_input(blend)
        if cleaned_blend is None:
            st.error("Invalid input format detected for 'Blend'. Please enter a valid number.")
            st.stop()
        blend = cleaned_blend


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

        st.metric(
            "Predicted Agtron",
            f"{agtron_prediction:.1f}"
        )
with tab2:

    st.header("Roast Recommendation")
    st.markdown("Enter target color and roast parameters to find optimal temperature and quench adjustments.")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        rec_month = st.selectbox(
            "Month ",
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
            ],
            key="rec_month"
        )

        rec_blend = st.text_input(
            "Blend ",
            value="284",
            key="rec_blend"
        )

        rec_consumed = st.number_input(
            "Consumed Weight (lb) ",
            min_value=0.0,
            value=500.0,
            step=1.0,
            key="rec_consumed"
        )

    with rec_col2:
        rec_temp = st.number_input(
            "Roast Temperature (°F) ",
            min_value=300.0,
            value=440.0,
            step=1.0,
            key="rec_temp"
        )

        rec_quench = st.number_input(
            "Quench Applied (lb) ",
            min_value=0.0,
            value=73.0,
            step=1.0,
            key="rec_quench"
        )

        target_agtron = st.number_input(
            "Target Agtron",
            min_value=15.0,
            max_value=65.0,
            value=45.0,
            step=0.1,
            key="rec_target_agtron"
        )

    recommend_button = st.button(
        "Recommend Settings"
    )
    if recommend_button:
        cleaned_blend = clean_blend_input(rec_blend)
        if cleaned_blend is None:
            st.error("Invalid input format detected for 'Blend'. Please enter a valid number.")
            st.stop()
        rec_blend = cleaned_blend

        input_data = pd.DataFrame({
            "Consumed": [rec_consumed],
            "Temp": [rec_temp],
            "Quench": [rec_quench],
            "Month": [str(rec_month)],
            "Blend": [rec_blend]        
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

        # Vectorized candidate evaluation (keeping existing search ranges)
        temp_changes = list(range(-11, 10))
        quench_changes = list(range(-6, 5))

        grid = [
            (rec_temp + tc, rec_quench + qc)
            for tc in temp_changes
            for qc in quench_changes
        ]
        candidate_temps, candidate_quenches = zip(*grid)
        n_candidates = len(grid)

        candidate_df = pd.DataFrame({
            "Consumed": [rec_consumed] * n_candidates,
            "Temp": candidate_temps,
            "Quench": candidate_quenches,
            "Month": [str(rec_month)] * n_candidates,
            "Blend": [rec_blend] * n_candidates
        })

        candidate_predictions = agtron_model.predict(candidate_df)
        candidate_errors = [abs(target_agtron - pred) for pred in candidate_predictions]

        recommendation_df = pd.DataFrame({
            "Temp": candidate_temps,
            "Quench": candidate_quenches,
            "Expected Agtron": [round(float(p), 1) for p in candidate_predictions],
            "Error": [round(float(e), 1) for e in candidate_errors]
        })

        recommendation_df = recommendation_df.sort_values("Error").reset_index(drop=True)
        best_row = recommendation_df.iloc[0]

        best_temp = best_row["Temp"]
        best_quench = best_row["Quench"]
        best_prediction = best_row["Expected Agtron"]
        best_error = best_row["Error"]

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