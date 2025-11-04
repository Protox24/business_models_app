
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

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

def compute_price(pbase, demand_factor, event_factor, inv_ratio):
    # Formula from PDF: Pt = Pbase * (1 + 0.05*D + 0.03*E - 0.02*I)
    return round(pbase * (1 + 0.05 * demand_factor + 0.03 * event_factor - 0.02 * inv_ratio), 2)

def simulate_step(event_factor=0.0, external_noise=0.1):
    # Simulate orders for this timestep (one 'tick')
    st.session_state.step += 1
    timestamp = datetime.now().strftime("%H:%M:%S")
    orders = {}
    total_orders = 0

    # Demand generation: baseline from popularity + small noise + recent momentum from last few steps
    for name, info in st.session_state.drinks.items():
        pop = info["popularity"]
        # Poisson lambda proportional to popularity and (inventory remaining / initial)
        inv_ratio = info["inventory"] / max(info.get("initial_inventory", info["inventory"]), 1)
        lam = max(0.1, pop * (1 + (1 - inv_ratio)) * 2.0)  # baseline expected orders
        # Add small random spikes
        qty = np.random.poisson(max(0.1, lam * (1 + np.random.normal(0, external_noise))))
        qty = int(qty)
        orders[name] = qty
        total_orders += qty

    # Update inventories and compute demand factor D (normalize by recent average)
    demand_factors = {}
    recent_window = 5
    for name, qty in orders.items():
        info = st.session_state.drinks[name]
        if "initial_inventory" not in info:
            info["initial_inventory"] = info["inventory"]
        info["inventory"] = max(0, info["inventory"] - qty)
        # demand factor: current orders normalized by a baseline (popularity*2)
        D = qty / (max(1, info["popularity"] * 2))
        demand_factors[name] = D

    # Update prices
    prices = {}
    for name, info in st.session_state.drinks.items():
        inv_ratio = info["inventory"] / max(info.get("initial_inventory", info["inventory"]), 1)
        D = demand_factors.get(name, 0.0)
        pt = compute_price(info["base_price"], D, event_factor, inv_ratio)
        prices[name] = pt
        st.session_state.price_history[name].append(pt)
    st.session_state.time_index.append(timestamp)

    # Record order history row
    row = {"time": timestamp}
    row.update(orders)
    row["total_orders"] = total_orders
    st.session_state.order_history.append(row)

    # Determine hourly/top buyer placeholder
    # (for prototype we use highest orders in this timestep)
    if total_orders > 0:
        top_drink = max(orders.items(), key=lambda x: x[1])[0]
        st.session_state.last_top_buyer = f"{top_drink} (qty {orders[top_drink]})"
    else:
        st.session_state.last_top_buyer = "—"

    return prices, orders

# --- Initialize state ---
init_state()

# --- Sidebar controls ---
st.sidebar.header("Controls")
event_factor = st.sidebar.slider("Event factor (E) — e.g., nearby event intensity", 0.0, 3.0, 0.0, 0.1)
external_noise = st.sidebar.slider("External randomness (noise)", 0.0, 1.0, 0.15, 0.01)
simulate_steps = st.sidebar.number_input("Simulate steps (each step = 1 tick)", min_value=1, max_value=500, value=1, step=1)
reset_btn = st.sidebar.button("Reset simulation")
add_orders = st.sidebar.button("Simulate single step now")

if reset_btn:
    st.session_state.drinks = default_drinks()
    st.session_state.time_index = []
    st.session_state.price_history = {k: [] for k in st.session_state.drinks.keys()}
    st.session_state.order_history = []
    st.session_state.step = 0
    st.session_state.last_top_buyer = None
    st.experimental_rerun()

# --- Top row: prices and metrics ---
st.title("SipSync — Prototipo de Engine de Precios Dinámicos")
col1, col2 = st.columns([3,2])

with col1:
    st.subheader("Mercado en vivo — Precios actuales")
    # show current prices (last price if exists otherwise base_price)
    current_prices = {}
    for name, info in st.session_state.drinks.items():
        ph = st.session_state.price_history.get(name, [])
        if ph:
            current_prices[name] = ph[-1]
        else:
            current_prices[name] = info["base_price"]

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

# --- Simulation controls in main area ---
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
    st.info("Aún no hay datos: presiona 'Simular N pasos' o 'Simular single step' en la barra lateral.")

st.markdown("### Historial de órdenes (últimos 50 registros)")
if st.session_state.order_history:
    df_orders = pd.DataFrame(st.session_state.order_history).fillna(0).tail(50)
    st.dataframe(df_orders)
else:
    st.write("Sin órdenes aún.")

st.markdown("### Parámetros y lógica")
st.write("Fórmula usada: Pt = Pbase × (1 + 0.05×D + 0.03×E − 0.02×I)")
st.write("D = factor de demanda (órdenes actuales normalizadas). E = factor de evento. I = inventario relativo (0 a 1).")

st.markdown("---")
st.caption("Prototipo educativo. No conectado a POS real. Ajusta parámetros para ver diferentes comportamientos.")
