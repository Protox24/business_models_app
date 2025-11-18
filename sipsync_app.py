import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="SipSync Auto Calibration", layout="wide")

# --- Helper functions ---
def default_drinks():
    return {
        "Mojito": {"base_price": 9.0, "inventory": 100, "initial_inventory": 100, "popularity": 1.0},
        "Gin Tonic": {"base_price": 6.0, "inventory": 150, "initial_inventory": 150, "popularity": 0.6},
        "Tequila Shot": {"base_price": 4.5, "inventory": 120, "initial_inventory": 120, "popularity": 0.8},
        "Spritz": {"base_price": 8.0, "inventory": 90,  "initial_inventory": 90,  "popularity": 0.7},
        "Beer": {"base_price": 3.5, "inventory": 200, "initial_inventory": 200, "popularity": 1.2},
    }


def compute_price_linear(pbase, D, E, inv_ratio, alpha, beta, gamma):
    I = 1 - inv_ratio
    return pbase * (1 + alpha * D + beta * E - gamma * I)


def simulate_step(drinks, alpha, beta, gamma, event_factor=0.0, external_noise=0.1):
    orders = {}
    DI_records = []

    for name, info in drinks.items():
        pop = info["popularity"]
        inv_ratio = info["inventory"] / max(info["initial_inventory"], 1)
        lam = max(0.1, pop * (1 + (1 - inv_ratio)) * 2.0)
        lam_noisy = max(0.1, lam * (1 + np.random.normal(0, external_noise)))
        qty = int(np.random.poisson(lam_noisy))
        orders[name] = qty
        info["inventory"] = max(0, info["inventory"] - qty)

        pt = compute_price_linear(info["base_price"], qty / max(1, pop*2), event_factor, inv_ratio, alpha, beta, gamma)

        DI_records.append({
            "drink": name,
            "p_base": info["base_price"],
            "p_t": pt,
            "D": qty / max(1, pop*2),
            "E": event_factor,
            "I": 1 - inv_ratio
        })

    return DI_records


def generate_dataset(alpha=0.4, beta=0.5, gamma=0.1, steps=500):
    drinks = default_drinks()
    all_records = []

    for _ in range(steps):
        # variar factor de evento aleatoriamente para diversificar la base
        event_factor = np.random.uniform(0, 3)
        records = simulate_step(drinks, alpha, beta, gamma, event_factor, external_noise=0.1)
        all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    return df


def estimate_coeffs(df):
    df = df.dropna()
    if df.empty:
        return None
    y = df["p_t"] / df["p_base"] - 1
    X = df[["D", "E", "I"]]
    model = LinearRegression()
    model.fit(X, y)
    return {
        "alpha": model.coef_[0],
        "beta": model.coef_[1],
        "gamma": -model.coef_[2],  # revertimos signo para interpretacion
        "intercept": model.intercept_,
        "r2": model.score(X, y),
        "n": len(df)
    }


# --- Streamlit app ---
st.title("SipSync Auto Calibration")

steps = st.number_input("Number of simulation steps", min_value=100, max_value=5000, value=500, step=100)

if st.button("Run Simulation & Estimate Coefficients"):
    with st.spinner("Simulating and calibrating..."):
        df = generate_dataset(steps=steps)
        coeffs = estimate_coeffs(df)

    st.success("Simulation completed!")
    st.write("### Estimated Coefficients")
    if coeffs:
        st.write(f"α (demand factor): {coeffs['alpha']:.4f}")
        st.write(f"β (event factor): {coeffs['beta']:.4f}")
        st.write(f"γ (inventory factor): {coeffs['gamma']:.4f}")
        st.write(f"Intercept: {coeffs['intercept']:.4f}")
        st.write(f"R²: {coeffs['r2']:.3f}, N={coeffs['n']}")
    else:
        st.warning("No se pudieron estimar coeficientes, genera más pasos o revisa los datos.")

    st.write("### Sample of simulation dataset")
    st.dataframe(df.head(20))

