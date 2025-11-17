import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="SipSync Prototype", layout="wide")

# --- Helper functions ---
def default_drinks():
    return {
        "Mojito": {"base_price": 9.0, "inventory": 100, "popularity": 1.0},
        "Gin Tonic": {"base_price": 6.0, "inventory": 150, "popularity": 0.6},
        "Tequila Shot": {"base_price": 4.5, "inventory": 120, "popularity": 0.8},
        "Spritz": {"base_price": 8.0, "inventory": 90, "popularity": 0.7},
        "Beer": {"base_price": 3.5, "inventory": 200, "popularity": 1.2},
    }

def init_state():
    if "drinks" not in st.session_state:
        st.session_state.drinks = default_drinks()
    if "time_index" not in st.session_state:
        st.session_state.time_index = []
    if "price_history" not in st.session_state:
        st.session_state.price_history = {k: [] for k in st.session_state.drinks.keys()}
    if "order_history" not in st.session_state:
        st.session_state.order_history = []
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "last_top_buyer" not in st.session_state:
        st.session_state.last_top_buyer = None
    if "DI_history" not in st.session_state:
        st.session_state.DI_history = {k: [] for k in st.session_state.drinks.keys()}
    # coeficientes globales (se usan en compute_price_linear)
    if "alpha" not in st.session_state:
        st.session_state.alpha = 0.05
    if "beta" not in st.session_state:
        st.session_state.beta = 0.03
    if "gamma" not in st.session_state:
        st.session_state.gamma = 0.02
    if "use_estimated_coeffs" not in st.session_state:
        st.session_state.use_estimated_coeffs = False

def compute_price_linear(pbase, D, E, inv_ratio,
                         alpha=None, beta=None, gamma=None,
                         max_up_pct=0.5, max_down_pct=0.3, min_step=0.1):
    """
    Fórmula lineal:
        P_t = Pbase × (1 + αD + βE − γI)
    donde I = (1 - inv_ratio)
    """
    # Usa coeficientes del estado si no se pasan explícitos
    if alpha is None: alpha = st.session_state.alpha
    if beta  is None: beta  = st.session_state.beta
    if gamma is None: gamma = st.session_state.gamma

    I = 1 - inv_ratio
    price_multiplier = 1 + alpha * D + beta * E - gamma * I
    raw_price = pbase * price_multiplier

    # Límites
    min_price = pbase * (1 - max_down_pct)
    max_price = pbase * (1 + max_up_pct)
    clipped_price = max(min(raw_price, max_price), min_price)

    # Redondeo
    rounded_price = round(clipped_price / min_step) * min_step
    return round(rounded_price, 2)

def simulate_step(event_factor=0.0, external_noise=0.1):
    st.session_state.step += 1
    timestamp = datetime.now().strftime("%H:%M:%S")
    orders = {}
    total_orders = 0

    # Generación de órdenes
    for name, info in st.session_state.drinks.items():
        pop = info["popularity"]
        inv_ratio = info["inventory"] / max(info.get("initial_inventory", info["inventory"]), 1)
        lam = max(0.1, pop * (1 + (1 - inv_ratio)) * 2.0)
        qty = np.random.poisson(max(0.1, lam * (1 + np.random.normal(0, external_noise))))
        qty = int(qty)
        orders[name] = qty
        total_orders += qty

    # Actualiza inventario y calcula D
    demand_factors = {}
    for name, qty in orders.items():
        info = st.session_state.drinks[name]
        if "initial_inventory" not in info:
            info["initial_inventory"] = info["inventory"]
        info["inventory"] = max(0, info["inventory"] - qty)
        D = qty / (max(1, info["popularity"] * 2))
        demand_factors[name] = D

    # Actualiza precios y guarda DI para regresión
    prices = {}
    for name, info in st.session_state.drinks.items():
        inv_ratio = info["inventory"] / max(info.get("initial_inventory", info["inventory"]), 1)
        D = demand_factors.get(name, 0.0)
        pt = compute_price_linear(info["base_price"], D, event_factor, inv_ratio)
        prices[name] = pt
        st.session_state.price_history[name].append(pt)

        st.session_state.DI_history[name].append({
            "time": timestamp,
            "drink": name,
            "p_base": info["base_price"],
            "p_t": pt,
            "D": D,
            "E": event_factor,
            "I": 1 - inv_ratio
        })

    st.session_state.time_index.append(timestamp)

    # Historial de órdenes
    row = {"time": timestamp}
    row.update(orders)
    row["total_orders"] = total_orders
    st.session_state.order_history.append(row)

    # Top buyer mock
    if total_orders > 0:
        top_drink = max(orders.items(), key=lambda x: x[1])[0]
        st.session_state.last_top_buyer = f"{top_drink} (qty {orders[top_drink]})"
    else:
        st.session_state.last_top_buyer = "—"

    return prices, orders

def estimate_coeffs(drink_name):
    df = pd.DataFrame(st.session_state.DI_history[drink_name])
    if df.empty:
        return None
    # Objetivo proporcional al precio base
    y = (df["p_t"] / df["p_base"]) - 1.0
    X = df[["D", "E", "I"]].copy()
    # En el modelo el término de I entra con signo negativo, aquí lo aprendemos y luego reportamos gamma positiva
    model = LinearRegression()
    model.fit(X, y)
    alpha = model.coef_[0]
    beta  = model.coef_[1]
    gamma_report = -model.coef_[2]  # revertimos signo para la interpretación
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma_report,
        "intercept": model.intercept_,
        "r2": model.score(X, y),
        "n": len(df)
    }

# --- Initialize state ---
init_state()

# --- Sidebar controls ---
st.sidebar.header("Controls")
event_factor = st.sidebar.slider("Event factor (E)", 0.0, 3.0, 0.0, 0.1)
external_noise = st.sidebar.slider("External randomness (noise)", 0.0, 1.0, 0.15, 0.01)
simulate_steps = st.sidebar.number_input("Simulate steps", min_value=1, max_value=500, value=1, step=1)
reset_btn = st.sidebar.button("Reset simulation")
add_orders = st.sidebar.button("Simulate single step now")

# Sliders para ajustar manualmente α, β, γ
st.sidebar.markdown("---")
st.sidebar.subheader("Coeficientes (α, β, γ)")
st.session_state.alpha = st.sidebar.slider("Alpha (demanda)", 0.0, 0.2, st.session_state.alpha, 0.005)
st.session_state.beta  = st.sidebar.slider("Beta (evento)",  0.0, 0.2, st.session_state.beta, 0.005)
st.session_state.gamma = st.sidebar.slider("Gamma (inventario)", 0.0, 0.2, st.session_state.gamma, 0.005)

if reset_btn:
    st.session_state.drinks = default_drinks()
    st.session_state.time_index = []
    st.session_state.price_history = {k: [] for k in st.session_state.drinks.keys()}
    st.session_state.order_history = []
    st.session_state.step = 0
    st.session_state.last_top_buyer = None
    st.session_state.DI_history = {k: [] for k in st.session_state.drinks.keys()}
    st.session_state.use_estimated_coeffs = False
    st.experimental_rerun()

# --- Top row: prices and metrics ---
st.title("SipSync — Prototipo de Engine de Precios Dinámicos (Modelo Lineal)")
col1, col2 = st.columns([3,2])

with col1:
    st.subheader("Mercado en vivo — Precios actuales")
    current_prices = {}
    for name, info in st.session_state.drinks.items():
        ph = st.session_state.price_history.get(name, [])
        current_prices[name] = ph[-1] if ph else info["base_price"]

    price_cols = st.columns(len(current_prices))
    for i, (name, price) in enumerate(current_prices.items()):
        with price_cols[i]:
            st.metric(label=name, value=f"€{price}")

with col2:
    st.subheader("Resumen rápido")
    st.write(f"Ticks simulados: {st.session_state.step}")
    st.write(f"Último 'top buyer': {st.session_state.last_top_buyer or '—'}")
    st.write("Inventario (restante):")
    for name, info in st.session_state.drinks.items():
        st.write(f"- {name}: {info['inventory']}")

# --- Simulation controls ---
st.markdown("---")
st.subheader("Simulación")

if st.button("Simular N pasos"):
    for _ in range(simulate_steps):
        simulate_step(event_factor=event_factor, external_noise=external_noise)

if add_orders:
    simulate_step(event_factor=event_factor, external_noise=external_noise)

# --- Charts ---
st.markdown("### Evolución de precios")
if st.session_state.time_index:
    df_prices = pd.DataFrame(st.session_state.price_history, index=st.session_state.time_index)
    st.line_chart(df_prices)
else:
    st.info("Aún no hay datos: presiona 'Simular N pasos' o 'Simular single step'.")

st.markdown("### Historial de órdenes (últimos 50 registros)")
if st.session_state.order_history:
    df_orders = pd.DataFrame(st.session_state.order_history).fillna(0).tail(50)
    st.dataframe(df_orders)
else:
    st.write("Sin órdenes aún.")

# --- Estimación de coeficientes ---
st.markdown("### Estimación de coeficientes α, β, γ")
drink_choice = st.selectbox("Selecciona bebida para calibrar:", list(st.session_state.drinks.keys()))
colA, colB = st.columns([1,1])

with colA:
    if st.button("Calcular coeficientes (regresión)"):
        coeffs = estimate_coeffs(drink_choice)
        if coeffs:
            st.session_state.estimated_coeffs = coeffs
            st.success(
                f"{drink_choice}: α={coeffs['alpha']:.3f}, β={coeffs['beta']:.3f}, γ={coeffs['gamma']:.3f} "
                f"(R²={coeffs['r2']:.2f}, n={coeffs['n']})"
            )
        else:
            st.warning("No hay suficientes datos para estimar coeficientes.")

with colB:
    if st.button("Aplicar coeficientes estimados globalmente"):
        coeffs = st.session_state.get("estimated_coeffs", None)
        if coeffs:
            st.session_state.alpha = float(coeffs["alpha"])
            st.session_state.beta  = float(coeffs["beta"])
            st.session_state.gamma = float(coeffs["gamma"])
            st.session_state.use_estimated_coeffs = True
            st.success(
                f"Aplicados: α={st.session_state.alpha:.3f}, β={st.session_state.beta:.3f}, γ={st.session_state.gamma:.3f}"
            )
        else:
            st.info("Primero calcula coeficientes con el botón de la izquierda.")

# Info de la fórmula
st.markdown("---")
st.markdown("#### Parámetros y lógica")
st.write("Fórmula usada: Pt = Pbase × (1 + α·D + β·E − γ·I)")
st.write("D = factor de demanda normalizado; E = factor de evento; I = inventario invertido (1 − stock/stock_inicial).")
st.caption("Prototipo educativo. No conectado a POS real. Ajusta parámetros para ver diferentes comportamientos.")

