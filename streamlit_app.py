import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- Load trained pipeline model ---
model_path = "output/random_forest_model.pkl"
model = joblib.load(model_path)

# --- Load dataset (for price search) ---
data_path = "output/engineered_scrape_cars_com.csv"
df_full = pd.read_csv(data_path)

# --- Page Settings ---
st.set_page_config(page_title="Car Price Estimator", layout="wide")
st.title("🚘 Car Price Estimator")
st.markdown("Estimate the **expected price** of a car based on its specifications.")

# --- User Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", ["BMW", "Audi", "Chevrolet", "Toyota", "Honda", "Ford", "Tesla", "Hyundai", "Kia", "Nissan"])
        fuel = st.selectbox("Fuel Type", ["Gasoline", "Hybrid", "Electric", "Diesel"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
        drivetrain = st.selectbox("Drivetrain", ["AWD", "4WD", "FWD", "RWD"])
        year = st.slider("Year", 2000, 2024, 2023)

    with col2:
        mileage = st.slider("Mileage (mi)", 0, 100000, 5000, step=1000)
        mpg = st.slider("Fuel Efficiency (MPG)", 5.0, 80.0, 25.0)

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
        predicted_price = model.predict(input_df)[0]
        buffer_percentage = 0.05  # ±5%
        price_min_estimate = predicted_price * (1 - buffer_percentage)
        price_max_estimate = predicted_price * (1 + buffer_percentage)

        st.subheader("🔎 Prediction Result")
        st.success(f"💰 Estimated Price Range: **${price_min_estimate:,.0f} - ${price_max_estimate:,.0f}**")
        st.caption(f"🎯 Central Estimate: ${predicted_price:,.2f}")

        # --- Three Side-by-Side Charts ---
        col_a, col_b, col_c = st.columns(3)

        # 1️⃣ Gauge Chart for Price
        with col_a:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_price,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Estimated Price ($)"},
                gauge={
                    "axis": {"range": [0, df_full["Price"].max()]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, df_full["Price"].median()], "color": "lightgray"},
                        {"range": [df_full["Price"].median(), df_full["Price"].max()], "color": "lightgreen"},
                    ],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 2️⃣ Radar Chart for Inputs
        with col_b:
            features = ["Mileage", "MPG", "Year", "Engine Size (L)"]
            values = [mileage, mpg, year, engine_size]

            bounds = {
                "Mileage": (0, 200000),
                "MPG": (5, 80),
                "Year": (2000, 2025),
                "Engine Size (L)": (0, 6)
            }

            normalized = [(v - bounds[f][0]) / (bounds[f][1] - bounds[f][0]) for f, v in zip(features, values)]

            radar_df = pd.DataFrame({
                "Feature": features + [features[0]],
                "Normalized": normalized + [normalized[0]]
            })

            fig_radar = px.line_polar(
                radar_df,
                r="Normalized",
                theta="Feature",
                line_close=True,
                title="Your Car's Profile",
                range_r=[0, 1]
            )
            fig_radar.update_traces(fill="toself")
            st.plotly_chart(fig_radar, use_container_width=True)

        # 3️⃣ Feature Importance
        with col_c:
            try:
                base_model = model.named_steps.get("regressor") or model.named_steps.get("model") or model
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

                        fig_bar = px.bar(
                            df_feat.head(10),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            color="Importance",
                            color_continuous_scale="Blues"
                        )
                        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("ℹ️ No matched features.")
                else:
                    st.info("ℹ️ Feature importance unavailable.")
            except Exception as e:
                st.warning(f"⚠️ Feature importance plot error: {e}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# --- Price Range Recommendation Feature ---
st.header("🎯 Best Cars within Your Estimated Price Range")

if submitted:
    # Filter the cars within predicted personalized range
    df_recommend = df_full[
        (df_full["Price"] >= price_min_estimate) &
        (df_full["Price"] <= price_max_estimate)
    ]

    if not df_recommend.empty:
        st.subheader("🏎️ 5 Best Cars by Sub-Price Categories")
        
        # Split into 5 price ranges
        price_bins = np.linspace(price_min_estimate, price_max_estimate, 6)
        df_recommend['Price Bin'] = pd.cut(df_recommend["Price"], bins=price_bins, labels=[1,2,3,4,5])

        final_selection = df_recommend.sort_values(
            ["Price Bin", "MPG", "Mileage", "Year"],
            ascending=[True, False, True, False]
        ).groupby("Price Bin").head(1)

        if not final_selection.empty:
            final_selection["Price"] = final_selection["Price"].apply(lambda x: f"${x:,.2f}")
            final_selection["Mileage"] = final_selection["Mileage"].apply(lambda x: f"{x:,.1f} mi")
            final_selection["Average MPG"] = final_selection["MPG"].apply(lambda x: f"{int(round(x))} MPG")

            display_cols = ["Title", "Price", "Mileage", "Average MPG", "Fuel Type", "Transmission", "Drivetrain"]
            final_selection = final_selection[display_cols].reset_index(drop=True)
            final_selection.index = np.arange(1, len(final_selection) + 1)
            st.table(final_selection)
        else:
            st.info("🔍 No cars found matching across sub-ranges.")
    else:
        st.info("🔍 No matching cars found in estimated range.")
