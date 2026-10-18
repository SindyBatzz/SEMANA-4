"""
Laboratorio de Recomendación — App de Streamlit (Grupo 10)
Construida a partir de streamlit_starter.py.

Permite a alguien de negocio:
  1. Elegir un cliente y un método, y ver sus 3 recomendaciones con el "porque".
  2. Comparar los métodos (hit rate@3, precisión@3 y cobertura) en el grupo control.
  3. Ver el resultado del experimento A/B y la lectura de negocio.

Correr con:  streamlit run streamlit_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"

# Ruta relativa al archivo: funciona igual en local, en Colab y en Streamlit Community Cloud
DATA = Path(__file__).parent / "data"

METODOS = ["Popularidad", "Filtrado por ítem", "Filtrado por cliente", "SVD (8 factores)", "Final (ítem + respaldo)"]


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
@st.cache_data
def cargar():
    catalogo = pd.read_csv(DATA / "catalogo.csv")
    hist = pd.read_csv(DATA / "ordenes_historial.csv")
    hold = pd.read_csv(DATA / "ordenes_holdout.csv")
    exp = pd.read_csv(DATA / "experimento.csv")
    clientes = pd.read_csv(DATA / "clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values

    # Ingreso por cliente en los 90 días (para el resumen del experimento A/B)
    precio = catalogo.set_index("item")["precio"]
    margen = catalogo.set_index("item")["margen"]
    hold["ingreso"] = hold["item"].map(precio)
    hold["margen_usd"] = hold["ingreso"] * hold["item"].map(margen)
    por_cliente = hold.groupby("customer_id")[["ingreso", "margen_usd"]].sum().reindex(M.index).fillna(0)
    ab = exp.set_index("customer_id").join(por_cliente)
    return catalogo, productos, M.index.values, X, NUEVO, es_control, ab


catalogo, productos, ids, X, NUEVO, es_control, ab = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)
tiene_hist = X.sum(axis=1) > 0


# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
@st.cache_data
def similitud_items(X):
    S = cosine_similarity(X.T)
    np.fill_diagonal(S, 0)
    return S


@st.cache_data
def modelo_svd(X, k=8):
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z = svd.fit_transform(X)
    return Z @ svd.components_


S_it = similitud_items(X)
S_svd_todos = modelo_svd(X)


def puntajes(metodo, x_vec, K, fila=None):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    sin_historial = x_vec.sum() == 0
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Final (ítem + respaldo)":
        # ítem para quien tiene historial; popularidad para el arranque en frío
        return popularidad.astype(float) if sin_historial else x_vec @ S_it
    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return x_vec @ S_it
    if metodo == "SVD (8 factores)":
        return S_svd_todos[fila]
    # TODO 3: filtrado por cliente: similitud con TODOS los demás, sus K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    sims = cosine_similarity(x_vec[None, :], X)[0]
    if fila is not None:
        sims[fila] = 0                                    # no contarse a sí mismo
    vec = np.argsort(-sims)[:K]
    return sims[vec] @ X[vec]


def vecinos(x_vec, K, fila):
    sims = cosine_similarity(x_vec[None, :], X)[0]
    sims[fila] = 0
    return np.argsort(-sims)[:K]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


def porque(metodo, j, x_vec, K, fila):
    """Explicación en lenguaje de negocio para el producto j."""
    sin_historial = x_vec.sum() == 0
    if metodo == "Popularidad" or (metodo == "Final (ítem + respaldo)" and sin_historial):
        rank = int((popularidad > popularidad[j]).sum()) + 1
        extra = " (cliente sin historial → respaldo de popularidad)" if metodo.startswith("Final") else ""
        return f"Es de lo más comprado: #{rank} del catálogo, {int(popularidad[j]):,} clientes lo han pedido{extra}"
    if metodo in ("Filtrado por ítem", "Final (ítem + respaldo)"):
        comprados = np.where(x_vec > 0)[0]
        mejor = comprados[np.argmax(S_it[comprados, j])]
        return f"Porque compró **{productos[mejor]}** (similitud {S_it[mejor, j]:.2f})"
    if metodo == "Filtrado por cliente":
        vec = vecinos(x_vec, K, fila)
        n = int(X[vec, j].sum())
        return f"{n} de sus {K} clientes más parecidos lo compraron"
    # SVD
    return "Encaja con el perfil de gustos del cliente (factores latentes del SVD)"


# ----------------------------------------------------------------------------------------------
# Evaluación (para la pestaña de comparación)
# ----------------------------------------------------------------------------------------------
@st.cache_data
def puntajes_todos(metodo, K, X):
    if metodo == "Popularidad":
        return np.tile(popularidad, (len(X), 1)).astype(float)
    if metodo == "Filtrado por ítem":
        return X @ S_it
    if metodo == "SVD (8 factores)":
        return S_svd_todos
    if metodo == "Final (ítem + respaldo)":
        return np.where(tiene_hist[:, None], X @ S_it, np.tile(popularidad, (len(X), 1)))
    # Filtrado por cliente, en bloques para no usar demasiada memoria
    S = np.zeros_like(X)
    for ini in range(0, len(X), 1000):
        sims = cosine_similarity(X[ini:ini + 1000], X)
        for r in range(sims.shape[0]):
            sims[r, ini + r] = 0
        vec = np.argsort(-sims, axis=1)[:, :K]
        w = np.take_along_axis(sims, vec, axis=1)
        S[ini:ini + 1000] = np.einsum("ik,ikj->ij", w, X[vec])
    return S


def top_k_todos(scores, K=3):
    s = np.where(X > 0, -np.inf, scores).astype(np.float64)
    s[:, COL_AGUA] = -np.inf
    return np.argsort(-s, axis=1)[:, :K]


@st.cache_data
def evaluar(metodos, K, n_eval, semilla=42):
    elegibles = np.where(es_control & (NUEVO.sum(axis=1) > 0))[0]
    rng = np.random.default_rng(semilla)
    muestra = rng.choice(elegibles, size=min(n_eval, len(elegibles)), replace=False)
    filas = {}
    for m in metodos:
        top = top_k_todos(puntajes_todos(m, K, X))
        aciertos = NUEVO[muestra[:, None], top[muestra]]
        filas[m] = {
            "Hit rate@3": aciertos.sum(axis=1).astype(bool).mean(),
            "Precisión@3": aciertos.sum(axis=1).mean() / 3,
            "Cobertura (de 28)": len(np.unique(top)),
        }
    return pd.DataFrame(filas).T, len(muestra)


# ----------------------------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------------------------
st.title("☕ Recomendador de productos — carrusel “Para ti”")
st.caption("Ayuda a decidir **qué recomendador** poner en el carrusel de la app y a **quién** mostrárselo, "
           "comparando qué tanto acierta cada método y cuánto ingreso extra genera el carrusel.")

# TODO 4: barra lateral con los controles interactivos
with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox("Método de recomendación", METODOS, index=4)
    K = st.slider("Vecinos K (filtrado por cliente)", min_value=5, max_value=200, value=30, step=5)
    st.divider()
    st.markdown("**Recomendación del Grupo 10:** _Final (ítem + respaldo)_ — ítem para clientes con historial, "
                "popularidad para clientes nuevos.")

tab1, tab2, tab3 = st.tabs(["🔎 Recomendar", "📊 Comparar métodos", "🧪 Experimento A/B"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
        if st.button("🎲 Cliente al azar"):
            st.session_state["cid_azar"] = int(np.random.choice(ids))
        if "cid_azar" in st.session_state:
            cid = st.session_state["cid_azar"]
            st.caption(f"Mostrando cliente al azar: {cid}")
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    comprados = [productos[j] for j in np.where(x_vec > 0)[0]]
    grupo = "Control" if es_control[fila] else "Tratamiento"

    with c2:
        st.markdown(f"**Cliente {cid}** · grupo del experimento: {grupo} · {len(comprados)} productos distintos en su historial")
        st.write("Ya compró:", ", ".join(comprados) if comprados else "— (sin historial)")

    # TODO 5: las 3 recomendaciones con su puntaje y su "porque"
    if not comprados and metodo in ("Filtrado por ítem", "Filtrado por cliente", "SVD (8 factores)"):
        st.warning("Este cliente **no tiene historial**: los métodos colaborativos le dan puntaje 0 a todo "
                   "y sus recomendaciones no significan nada. Usen **Popularidad** o **Final (ítem + respaldo)**.")

    scores = puntajes(metodo, x_vec, K, fila)
    top = top3(scores, x_vec)
    st.subheader(f"Top 3 · {metodo}")
    cols = st.columns(3)
    for pos, (col, j) in enumerate(zip(cols, top), start=1):
        with col:
            info = catalogo.iloc[j]
            st.metric(f"#{pos} · {productos[j]}", f"US${info['precio']:.2f}", f"puntaje {scores[j]:.2f}",
                      delta_color="off")
            st.markdown(porque(metodo, j, x_vec, K, fila))
            st.caption(f"Categoría: {info['categoria']}")

    if NUEVO[fila].sum() > 0:
        nuevos = [productos[j] for j in np.where(NUEVO[fila] > 0)[0]]
        acierto = any(NUEVO[fila, j] for j in top)
        st.info(f"En los 90 días siguientes probó por primera vez: {', '.join(nuevos)} → "
                + ("**¡la recomendación acertó!**" if acierto else "ninguna de las 3 recomendaciones acertó."))

with tab2:
    # TODO 6: tabla de comparación en el grupo CONTROL, con controles interactivos
    st.markdown("Evaluación con el **holdout de 90 días**, solo en clientes del **grupo control** que probaron "
                "algo nuevo (compras sin ninguna influencia del carrusel).")
    elegidos = st.multiselect("Métodos a comparar", METODOS, default=METODOS)
    n_eval = st.slider("¿Cuántos clientes evaluar?", 200, 1797, 1797, step=100,
                       help="Con menos clientes la tabla es más rápida pero más ruidosa.")
    if elegidos:
        with st.spinner("Calculando…"):
            tabla, n = evaluar(tuple(elegidos), K, n_eval)
        st.caption(f"{n:,} clientes evaluados · K = {K} vecinos para el filtrado por cliente")
        st.dataframe(tabla.style.format({"Hit rate@3": "{:.3f}", "Precisión@3": "{:.3f}",
                                         "Cobertura (de 28)": "{:.0f}"})
                     .highlight_max(subset=["Hit rate@3", "Precisión@3"], color="#cfe8dc"),
                     width="stretch")
        st.bar_chart(tabla["Hit rate@3"], color=VERDE, horizontal=True)

    st.subheader("Hit rate vs. ingreso incremental real")
    st.markdown("Resultado de la *revelación final* del laboratorio (información oculta del generador):")
    rev = pd.DataFrame({
        "Hit rate@3 (control)": [0.221, 0.401, 0.301, 0.369, 0.410],
        "Ingreso incremental (US$/cliente)": [1.00, 1.19, 1.19, 1.21, 1.18],
        "% de aciertos que ya iban a ocurrir": ["31%", "44%", "38%", "41%", "44%"],
    }, index=METODOS)
    st.dataframe(rev, width="stretch")
    st.warning("El filtrado por ítem acierta **81% más** que la popularidad, pero genera solo **19% más** ingreso "
               "incremental: casi la mitad de sus aciertos eran compras que **ya iban a ocurrir**. "
               "Un hit rate alto no equivale a ingreso extra.")

with tab3:
    st.markdown("Experimento de 90 días: la mitad de los clientes (al azar) vio el carrusel de **popularidad**.")
    costo = st.slider("Costo del carrusel por cliente tratado (US$)", 0.0, 1.0, 0.20, 0.05)
    r = st.slider("Retención trimestral r", 0.50, 0.95, 0.78, 0.01)
    d = 0.03
    trat, ctrl = ab[ab["grupo"] == "tratamiento"], ab[ab["grupo"] == "control"]
    d_arpu = trat["ingreso"].mean() - ctrl["ingreso"].mean()
    ee = np.sqrt(trat["ingreso"].var() / len(trat) + ctrl["ingreso"].var() / len(ctrl))
    d_margen_neto = trat["margen_usd"].mean() - ctrl["margen_usd"].mean() - costo
    factor = r / (1 + d - r)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ΔARPU (90 días)", f"US${d_arpu:+.2f}", f"IC 95% {d_arpu - 1.96 * ee:+.2f} a {d_arpu + 1.96 * ee:+.2f}",
              delta_color="off")
    m2.metric("Δmargen neto", f"US${d_margen_neto:+.2f}", "después del costo", delta_color="off")
    m3.metric("ΔCLV por cliente", f"US${d_margen_neto * factor:+.2f}", f"factor de vida {factor:.2f}",
              delta_color="off")
    m4.metric("Total (4,000 tratados)", f"US${d_margen_neto * factor * len(trat):+,.0f}")
    st.caption("CLV = margen × r / (1 + d − r), con d = 0.03 trimestral. Supuesto: el efecto del carrusel se "
               "mantiene mientras el cliente siga activo (el más frágil, por el sesgo de novedad).")
