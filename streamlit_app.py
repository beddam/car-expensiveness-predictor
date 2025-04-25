import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load trained pipeline model
model_path = "output/random_forest_model.pkl"
model = joblib.load(model_path)

st.set_page_config(page_title="Car Expensiveness Predictor", layout="centered")
st.title("🚘 Car Expensiveness Predictor")
st.markdown("Estimate whether a car is likely to be **in the top 25% of price** based on its specifications.")

# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", ["BMW", "Audi", "Chevrolet", "Toyota", "Honda", "Ford", "Tesla", "Hyundai", "Kia", "Nissan"])
        fuel = st.selectbox("Fuel Type", ["Gasoline", "Hybrid", "Electric", "Diesel"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Other"])
        drivetrain = st.selectbox("Drivetrain", ["AWD", "4WD", "FWD", "RWD"])
        year = st.slider("Year", 2000, 2024, 2023)

    with col2:
        mileage = st.slider("Mileage (mi)", 0, 300000, 5000, step=1000)
        mpg = st.slider("Fuel Efficiency (MPG)", 5.0, 150.0, 25.0)
        engine_size = st.slider("Engine Size (Liters)", 0.0, 8.0, 2.0, step=0.1)

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction Logic ---
if submitted:
    car_age = 2025 - year
    mpg_engine = mpg * engine_size
    is_electric = 1 if fuel.lower() == "electric" else 0
    is_hybrid = 1 if "hybrid" in fuel.lower() else 0
    is_gasoline = 1 if fuel.lower() == "gasoline" else 0
    is_diesel = 1 if fuel.lower() == "diesel" else 0
    is_new = 1 if car_age <= 1 else 0
    is_low_mileage = 1 if mileage < 20000 else 0
    log_mileage = np.log1p(mileage)

    # Input DataFrame
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Fuel Type": fuel,
        "Transmission": transmission,
        "Drivetrain": drivetrain,
        "Year": year,
        "Mileage": mileage,
        "MPG": mpg,
        "Engine Size (L)": engine_size,
        "Car Age": car_age,
        "MPG*Engine": mpg_engine,
        "IsElectric": is_electric,
        "IsHybrid": is_hybrid,
        "IsGasoline": is_gasoline,
        "IsDiesel": is_diesel,
        "IsNewCar": is_new,
        "IsLowMileage": is_low_mileage,
        "Log Mileage": log_mileage
    }])

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("🔎 Prediction Result")
        st.success("✅ Expensive" if prediction == 1 else "❌ Not Expensive")
        st.markdown(f"**Confidence Score:** {proba*100:.2f}%")

        # Show only real input features in top influences
        try:
            base_model = model.named_steps.get("classifier") or model.named_steps.get("model") or model
            if hasattr(base_model, "feature_importances_"):
                importances = base_model.feature_importances_
                feature_names = model.named_steps["preprocess"].get_feature_names_out()

                # Keep only raw input features (filtered by name)
                allowed_features = [
                    "Brand", "Fuel Type", "Transmission", "Drivetrain", "Year", "Mileage", "MPG",
                    "Engine Size (L)"
                ]

                # Match and clean feature names
                mapped_features = []
                for fname in feature_names:
                    clean = fname.split("__")[-1]
                    for af in allowed_features:
                        if af in clean:
                            mapped_features.append((af, importances[list(feature_names).index(fname)]))
                            break

                if mapped_features:
                    df_feat = pd.DataFrame(mapped_features, columns=["Feature", "Importance"])
                    df_feat = df_feat.groupby("Feature").sum().sort_values("Importance", ascending=False).reset_index()

                    st.subheader("📊 Most Influential Inputs")
                    fig = px.bar(
                        df_feat.head(10),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig)
                else:
                    st.info("ℹ️ No matched features to display.")
            else:
                st.info("ℹ️ Feature importance not available.")
        except Exception as e:
            st.warning(f"⚠️ Feature importance explanation failed: {e}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
