import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --- Configuration & Setup ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "engagement_model (2).pkl"

SENTIMENT_SCORES = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def prepare_features(
        platform: str,
        post_type: str,
        post_time: pd.Timestamp,
        sentiment: str,
) -> pd.DataFrame:
    hour = post_time.hour
    if 5 <= hour < 12:
        posting_period = "Morning"
    elif 12 <= hour < 17:
        posting_period = "Afternoon"
    elif 17 <= hour < 21:
        posting_period = "Evening"
    else:
        posting_period = "Night"

    return pd.DataFrame(
        [
            {
                "platform": platform,
                "post_type": post_type,
                "post_day": post_time.day_name(),
                "posting_period": posting_period,
                "posting_hour": hour,
                "is_weekend": int(post_time.weekday() >= 5),
                "sentiment_score": SENTIMENT_SCORES[sentiment],
            }
        ]
    )


# --- Streamlit UI ---
st.set_page_config(page_title="Engagement Predictor", layout="centered")
st.title("Social Media Engagement Prediction")

# Load model
model = load_model()
if model is None:
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

# --- Input Widgets ---
# Variables prefixed with 'selected_' prevent shadowing the function parameters
col1, col2 = st.columns(2)

with col1:
    selected_platform = st.selectbox("Platform", ["Twitter", "LinkedIn", "Instagram", "Facebook"])
    selected_post_type = st.selectbox("Post Type", ["Text", "Image", "Video", "Link"])
    selected_sentiment = st.selectbox("Sentiment", ["positive", "neutral", "negative"])

with col2:
    # Handling Date Input (Resolves type warning for potential tuple)
    raw_date = st.date_input("Post Date", value=datetime.date.today())
    if isinstance(raw_date, tuple):
        selected_date = raw_date[0] if len(raw_date) > 0 else datetime.date.today()
    else:
        selected_date = raw_date

    # Handling Time Input (Resolves None type warning)
    raw_time = st.time_input("Post Time", value=datetime.datetime.now().time())
    assert raw_time is not None
    selected_time = raw_time

# Combine date and time into a single Pandas Timestamp
assert isinstance(selected_date, datetime.date)
combined_datetime = datetime.datetime.combine(selected_date, selected_time)
selected_post_time = pd.Timestamp(combined_datetime)

# --- Prediction Logic ---
if st.button("Predict Engagement", type="primary"):
    try:
        # Prepare the features DataFrame
        features_df = prepare_features(
            platform=selected_platform,
            post_type=selected_post_type,
            post_time=selected_post_time,
            sentiment=selected_sentiment,
        )

        # Display the formatted input data for transparency
        st.write("**Processed Input Features:**")
        st.dataframe(features_df, hide_index=True)

        # Execute prediction
        from sklearn.pipeline import Pipeline  # Or whichever class your model is

        # Right before predicting:
        assert isinstance(model, Pipeline)
        prediction = model.predict(features_df)

        # Display result
        st.success(f"### Predicted Engagement: {prediction[0]}")

    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        st.info(
            "Ensure your saved `.pkl` model includes a preprocessing pipeline (like OneHotEncoder) to handle categorical strings such as 'platform' and 'post_type'.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")