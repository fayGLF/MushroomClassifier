import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("mushroom_model.pkl")
feature_names = list(model.feature_names_in_)

# Extract raw feature categories
raw_features = {
    "cap-shape": ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'],
    "cap-surface": ['fibrous', 'grooves', 'scaly', 'smooth'],
    "bruises": ['bruises', 'no'],
    "gill-spacing": ['close', 'crowded'],
    "gill-size": ['broad', 'narrow'],
    "gill-color": ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'],
    "stalk-root": ['bulbous', 'club', 'equal', 'rooted'],
    "stalk-surface-above-ring": ['fibrous', 'scaly', 'silky', 'smooth'],
    "stalk-surface-below-ring": ['fibrous', 'scaly', 'silky', 'smooth'],
    "stalk-color-above-ring": ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'],
    "stalk-color-below-ring": ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'],
    "veil-color": ['brown', 'orange', 'white', 'yellow'],
    "ring-number": ['none', 'one', 'two'],
    "ring-type": ['evanescent', 'flaring', 'large', 'none', 'pendant'],
    "spore-print-color": ['black', 'brown', 'buff', 'chocolate', 'orange', 'purple', 'white', 'yellow'],
    "population": ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'],
    "habitat": ['desert', 'grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste']
}

st.image("mushroom-banner.png", use_container_width=True)


st.write("""
Welcome! Select the mushroom traits below to check if it is **edible** or **poisonous**.
""")

st.image("mushroom-parts.png", use_container_width=True)

st.subheader("Choose Mushroom Traits")
cols = st.columns(3)
user_input = {}

# Collect inputs
for idx, (feature, options) in enumerate(raw_features.items()):
    with cols[idx % 3]:
        selected = st.selectbox(f"{feature.replace('-', ' ').capitalize()}", options)
        user_input[feature] = selected
       
# Prediction
if st.button("Predict!"):
    # Prepare input for model
    input_df = pd.DataFrame(columns=feature_names)
    input_df.loc[0] = 0

    for feature, value in user_input.items():
        col_name = f"{feature}_{value}"
        if col_name in input_df.columns:
            input_df.at[0, col_name] = 1

    # Make prediction
    prediction = model.predict(input_df)[0]
    proba_all = model.predict_proba(input_df)[0]
    confidence = proba_all[prediction]
    label = "poisonous" if prediction == 1 else "edible"

    # Result Display
    if confidence >= 0.80:
        if label == "poisonous":
            st.image("mushroom-poison.png", width=300)
            st.markdown(
                f"<h2 style='text-align:center; color:red;'>☠️ Definitely Poisonous!</h2>"
                f"<h3 style='text-align:center; color:black;'>Confidence: {confidence:.2%}</h3>",
                unsafe_allow_html=True
            )
        else:
            st.image("mushroom-edible.png", width=300)
            st.markdown(
                f"<h2 style='text-align:center; color:green;'>🍽️ Definitely Edible!</h2>"
                f"<h3 style='text-align:center; color:black;'>Confidence: {confidence:.2%}</h3>",
                unsafe_allow_html=True
            )
    else:
        st.image("mushroom-skate.png", width=300)
        st.markdown(
            f"<h2 style='text-align:center; color:orange;'>🛹 Uncertain!</h2>"
            f"<h3 style='text-align:center; color:black;'>Confidence: {confidence:.2%}</h3>",
            unsafe_allow_html=True
        )



st.image("mushroom-group.png", use_container_width=True)

# Sidebar info
with st.sidebar:
    st.title("Model Info")
    st.write("**Model:** Random Forest")
    st.write("**Accuracy:** 100% 🚀")
    st.write(f"**Features used:** 17")
    st.success("Built with Streamlit + scikit-learn")
