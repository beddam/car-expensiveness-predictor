import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load trained pipeline model
model_path = "output/random_forest_model.pkl"
model = joblib.load(model_path)

# Load the dataset (for price search feature)
data_path = "output/engineered_scrape_cars_com.csv"
df_full = pd.read_csv(data_path)

st.set_page_config(page_title="Car Expensiveness Predictor", layout="centered")
st.title("🚘 Car Expensiveness Predictor")
st.markdown("Estimate whether a car is likely to be **in the top 25% of price** based on its specifications.")

# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", ["BMW", "Audi", "Chevrolet", "Toyota", "Honda", "Ford", "Tesla", "Hyundai", "Kia", "Nissan"])
        fuel = st.selectbox("Fuel Type", ["Gasoline", "Hybrid", "Electric", "Diesel"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"])  # No 'Other'
        drivetrain = st.selectbox("Drivetrain", ["AWD", "4WD", "FWD", "RWD"])
        year = st.slider("Year", 2000, 2024, 2023)

    with col2:
        mileage = st.slider("Mileage (mi)", 0, 200000, 5000, step=1000)
        mpg = st.slider("Fuel Efficiency (MPG)", 5.0, 80.0, 25.0)

        # Disable Engine Size if Electric
        if fuel == "Electric":
            engine_size = 0.0
            st.caption("⚡ Electric selected — Engine Size set to 0.0L automatically.")
        else:
            engine_size = st.slider("Engine Size (Liters)", 0.0, 6.0, 2.0, step=0.1)

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction Logic ---
if submitted:
    car_age = 2025 - year
    mpg_engine = mpg * engine_size
    is_electric = 1 if fuel.lower() == "electric" else 0
    is_hybrid = 1 if "hybrid" in fuel.lower() else 0
    is_gasoline = 1 if fuel.lower() == "gasoline" else 0
    is_diesel = 1 if fuel.lower() == "diesel" else 0
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
        "MPG*Engine": mpg_engine,
        "Log Mileage": log_mileage,
        "IsElectric": is_electric,
        "IsHybrid": is_hybrid,
        "IsGasoline": is_gasoline,
        "IsDiesel": is_diesel,
        "IsLowMileage": is_low_mileage
    }])

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("🔎 Prediction Result")
        st.success("✅ Expensive" if prediction == 1 else "❌ Not Expensive")
        st.markdown(f"**Confidence Score:** {proba*100:.2f}%")

        # Show feature importance
        try:
            base_model = model.named_steps.get("classifier") or model.named_steps.get("model") or model
            if hasattr(base_model, "feature_importances_"):
                importances = base_model.feature_importances_
                feature_names = model.named_steps["preprocess"].get_feature_names_out()

                allowed_features = [
                    "Brand", "Fuel Type", "Transmission", "Drivetrain", "Year", "Mileage", "MPG", "Engine Size (L)"
                ]

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

# --- Price Range Recommendation Feature ---
st.header("🎯 Find Best Cars by Price")

# Use Range Slider
price_range = st.slider(
    "Select Price Range ($)",
    min_value=int(df_full["Price"].min()),
    max_value=int(df_full["Price"].max()),
    value=(10000, 40000),
    step=1000
)

search_submitted = st.button("🔎 Find Cars")

if search_submitted:
    price_min, price_max = price_range

    results = df_full[
        (df_full["Price"] >= price_min) & (df_full["Price"] <= price_max)
    ].sort_values(
        ["IsExpensive", "MPG", "Mileage"],
        ascending=[False, False, True]
    ).head(5)

    if not results.empty:
        st.subheader("🏎️ Top 5 Cars Matching Your Budget")

        # Format results
        results_display = results.copy()
        results_display["Price"] = results_display["Price"].apply(lambda x: f"${x:,.2f}")
        results_display["Mileage"] = results_display["Mileage"].apply(lambda x: f"{x:,.1f} mi")
        results_display["Average MPG"] = results_display["MPG"].apply(lambda x: f"{int(round(x))} MPG")
        
        display_cols = ["Title", "Price", "Mileage", "Average MPG", "Fuel Type", "Transmission", "Drivetrain"]

        results_display = results_display[display_cols].reset_index(drop=True)
        results_display.index = np.arange(1, len(results_display) + 1)
        st.table(results_display)

    else:
        st.info("🔍 No matching cars found in that range.")
