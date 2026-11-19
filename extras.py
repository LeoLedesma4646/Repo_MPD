"""Páginas añadidas a la app integrada (Grupo 5).

- Inicio y objetivos: objetivos específicos cuantificados con resultado en vivo.
- Datos y análisis descriptivo: estadísticos por variable, estandarización, correlación y PCA.
- Tablero BI integrador: réplica de las 4 páginas del reporte Power BI (PBIX).
- Comparar escenarios: evaluación prescriptiva de varios escenarios de precio y servicio.
- Fundamentación: algoritmos, regularización y referencias (se agrega a «Método y fuentes»).
Todo reutiliza el mismo motor (motor.py), los mismos modelos y los mismos artefactos.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import motor as M
from modelos import ROOT, fc, NAMES, predict

INK, BASE, SCEN = "#123b5d", "#3978b3", "#cf6225"
MODEL_COLORS = {"Realidad": "#5B6573", fc.MODEL_NAIVE: "#E66101", fc.MODEL_RIDGE: "#7B3FA0",
                fc.MODEL_SARIMAX: "#1F78B4", fc.MODEL_DHR: "#33A02C"}
STATUS_COLORS = {"Agotamiento crítico": "#D9534F", "Reordenar urgente": "#F0AD4E",
                 "Óptimo": "#5CB85C", "Sobre-stock": "#0275D8"}
STATUS_ORDER = list(STATUS_COLORS)
LAYOUT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
              margin=dict(l=40, r=30, t=60, b=90), font=dict(color="#172f46"),
              legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
NUMERIC_VARS = ["units_sold", "price_unit", "promotion_flag", "stock_available", "delivery_days",
                "is_holiday_peak", "is_holiday_week", "is_summer", "is_winter", "lag_1", "lag_2",
                "rolling_mean_4", "rolling_std_4", "momentum", "target_next_week"]
VAR_LABELS = {"units_sold": "Unidades vendidas", "price_unit": "Precio unitario", "promotion_flag": "Promoción (0/1)",
              "stock_available": "Stock disponible", "delivery_days": "Días de entrega",
              "is_holiday_peak": "Pico festivo (0/1)", "is_holiday_week": "Semana festiva (0/1)",
              "is_summer": "Verano (0/1)", "is_winter": "Invierno (0/1)", "lag_1": "Rezago 1 semana",
              "lag_2": "Rezago 2 semanas", "rolling_mean_4": "Media móvil 4 sem.",
              "rolling_std_4": "Desv. móvil 4 sem.", "momentum": "Momentum",
              "target_next_week": "Demanda semana siguiente"}


def plot(fig, **kw):
    st.plotly_chart(fig, theme=None, width="stretch", **kw)


def vline(fig, x, text=None, dash="dot"):
    if hasattr(x, "to_pydatetime"):
        x = x.to_pydatetime()
    fig.add_vline(x=x, line_dash=dash, line_color="#64748B")
    if text:
        fig.add_annotation(x=x, y=1, yref="paper", text=text, showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11, color="#475569"))


# ============================================================
# Datos compartidos (en caché)
# ============================================================
@st.cache_data(show_spinner="Cargando la base…")
def raw() -> pd.DataFrame:
    df = M.load_data()
    df["sku"] = df["sku"].astype(str)
    for c in NUMERIC_VARS:
        if c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def registry() -> pd.DataFrame:
    return M.load_registry()


@st.cache_data(show_spinner=False)
def priority_skus() -> list[str]:
    """10 SKU priorizados del informe: ≥104 semanas comunes y mayor ingreso histórico."""
    s = fc.sku_summary(raw())
    return fc.select_top_skus(s, 10)["sku"].tolist()


@st.cache_data(show_spinner=False)
def metrics() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts/metrics.csv")


@st.cache_data(show_spinner=False)
def elasticity() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "artifacts/elasticity.csv")
    p = pd.read_csv(ROOT / "artifacts/price_scenario.csv")[["sku", "elasticity_used", "scenario_elasticity_low",
                                                              "scenario_elasticity_high", "epsilon_corrected_free",
                                                              "corrected_ci_low", "corrected_ci_high"]]
    e = e.merge(p, on="sku", how="left")
    md = raw()
    md = md[(md.units_sold > 0) & (md.price_unit > 0)]
    g = md.groupby("sku").agg(unidades_totales=("units_sold", "sum"), precio_promedio=("price_unit", "mean")).reset_index()
    e = e.merge(g, on="sku", how="left")
    e["significativo_5pct"] = np.where(e["p_cluster"] < 0.05, "Sí", "No")
    e["clasificacion"] = np.where(e["elasticity"].abs() > 1, "Demanda elástica", "Demanda inelástica")
    return e


@st.cache_data(show_spinner="Calculando el forecast de los cuatro modelos…")
def all_models_forecast(horizon: int = 12) -> pd.DataFrame:
    """Forecast de los 4 modelos por SKU desde el corte global (equivalente a BI_Forecast)."""
    data = raw()
    man = M.load_manifest()
    as_of = pd.Timestamp(man["as_of"])
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sku in registry().sku:
            meta = M.load_meta(sku)
            hist = fc.aggregate_sku(data, sku)
            gap = int((as_of - hist.week.iloc[-1]).days // 7)
            fut = fc.future_scenario(hist, horizon + gap, source_df=data)
            fut["price_unit"] = float(hist.price_unit.tail(4).mean())
            fut["is_holiday_week"] = 0
            for name in NAMES:
                params = meta["params"].get(name, {})
                try:
                    model = M.load_model(sku, name, hist, params)
                    y = predict(name, model, hist, fut, params)
                except Exception:
                    continue
                rows.append(pd.DataFrame({"sku": sku, "week": fut.week.iloc[gap:].to_numpy(),
                                          "model": name, "value": y[gap:], "is_winner": name == meta["winner"]}))
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False)
def history(sku: str) -> pd.DataFrame:
    return fc.aggregate_sku(raw(), sku)


@st.cache_data(show_spinner="Calculando reposición base de los SKU…")
def default_inventory(service: float = 0.95) -> pd.DataFrame:
    """Política académica por defecto de la app (L=2, σL=0.5, R=1) sin cambio de precio."""
    rows = []
    data = raw()
    for sku in registry().sku:
        c = M.Scenario(sku=sku, service=service)
        r = M.calculate(c, M.forecast_base(c, data))
        b = r["inventory"].set_index("scenario").loc["Base"]
        rows.append({"sku": sku, "modelo_campeon": r["info"]["forecast_model"],
                     "forecast_12w": float(r["weekly"].base_demand.sum()),
                     "SS": float(b.safety_stock), "ROP": float(b.reorder_point),
                     "S": float(b.target_stock), "servicio_simulado": float(b.simulated_cycle_service)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Preparando datos del tablero de inventarios…")
def bi_inventory_rows() -> pd.DataFrame:
    """Réplica del script R de Power BI (Sofía), por fila canal-región-semana."""
    d = raw()[["sku", "week", "year", "units_sold", "stock_available", "delivery_days",
               "rolling_mean_4", "rolling_std_4"]].copy()
    z, R, sL = norm.ppf(0.95), 1, 0.1
    L = d["delivery_days"] / 7
    D = np.where(d["rolling_mean_4"].isna() | (d["rolling_mean_4"] == 0), d["units_sold"], d["rolling_mean_4"])
    sD = d["rolling_std_4"].fillna(0)
    d["Stock_Seguridad"] = np.ceil(z * np.sqrt(L * sD ** 2 + D ** 2 * sL ** 2))
    d["Punto_Reorden_ROP"] = np.ceil(D * L + d["Stock_Seguridad"])
    d["Inventario_Maximo_S"] = np.ceil(D * (L + R) + d["Stock_Seguridad"])
    return d


def classify(stock, ss, rop, smax):
    if stock < ss:
        return "Agotamiento crítico"
    if stock < rop:
        return "Reordenar urgente"
    if stock > smax:
        return "Sobre-stock"
    return "Óptimo"


@st.cache_data(show_spinner=False)
def bi_status_latest() -> pd.DataFrame:
    d = bi_inventory_rows()
    out = []
    for sku, g in d.groupby("sku"):
        last = g["week"].max()
        x = g[g["week"] == last]
        s = dict(stock=x.stock_available.sum(), ss=x.Stock_Seguridad.sum(), rop=x.Punto_Reorden_ROP.sum(),
                 smax=x.Inventario_Maximo_S.sum())
        out.append({"sku": sku, "semana": last, **s, "estado": classify(s["stock"], s["ss"], s["rop"], s["smax"])})
    return pd.DataFrame(out)


def describe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        mode = s.mode()
        rows.append({"Variable": c, "Descripción": VAR_LABELS.get(c, c), "n": int(s.size), "Media": s.mean(),
                     "Mediana": s.median(), "Moda": mode.iloc[0] if len(mode) else np.nan, "Desv. estándar": s.std(),
                     "Mínimo": s.min(), "Máximo": s.max(),
                     "Coef. variación (%)": 100 * s.std() / s.mean() if s.mean() else np.nan})
    return pd.DataFrame(rows)


# ============================================================
# Página: Inicio y objetivos
# ============================================================
def page_inicio():
    st.subheader("Modelamiento predictivo y analítico de ventas por SKU")
    st.caption("Universidad de Lima · Ingeniería Industrial · Modelamiento predictivo de datos · Sección 805 · "
               "Prof. José Antonio Taquia Gutiérrez · Grupo 5: Sofía Gamboa, Valeria Guzmán, Leonardo Ledesma y Cinzia Torres")
    reg = registry()
    el = elasticity()
    inv = default_inventory()
    mt = metrics()
    hold = mt[mt.scope.eq("holdout")]
    naive_hold = hold[hold.model.eq(fc.MODEL_NAIVE)].set_index("sku")["WAPE"]
    long = reg[reg.n_weeks >= 104]
    c = st.columns(5)
    c[0].metric("Registros", f"{len(raw()):,}".replace(",", " "))
    c[1].metric("SKU modelados", f"{len(reg)}")
    c[2].metric("WAPE comprobación (mediana)", f"{reg.holdout_WAPE.median():.1f}%")
    c[3].metric("Elasticidad OLS significativa", f"{int((el.p_cluster < .05).sum())} de {len(el)}")
    c[4].metric("Servicio simulado ≥ 95%", f"{int((inv.servicio_simulado >= .95).sum())} de {len(inv)}")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Problema")
        st.write("En retail, el error de pronóstico se traduce en exceso de inventario, quiebres de stock o ventas "
                 "perdidas; además, cada SKU responde distinto al precio y a las promociones. La solución conecta "
                 "en una sola herramienta cuánto se venderá, qué tan sensible es la demanda al precio, qué "
                 "inventario protege el servicio y cómo priorizar decisiones.")
        st.markdown("#### Objetivo general")
        st.write("Desarrollar una solución analítica integrada por SKU que pronostique la demanda semanal, estime la "
                 "sensibilidad precio-demanda, traduzca la demanda esperada en parámetros de reposición y consolide "
                 "los resultados en un tablero de soporte a decisiones.")
    with right:
        st.markdown("#### Arquitectura")
        st.graphviz_chart("""
digraph G { rankdir=TB; bgcolor="transparent"; nodesep=.35; ranksep=.4;
 node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#123b5d", fillcolor="#EEF3F8"];
 d [label="Base semanal · 30 SKU\\n3 canales · 3 regiones", fillcolor="white"];
 f [label="1 · Forecast\\nNaive · Fourier+Ridge · SARIMAX · DHR"];
 e [label="2 · Respuesta al precio\\nOLS log-log + modelo semanal"];
 i [label="3 · Inventarios\\nrevisión periódica · simulación"];
 bi [label="4 · Tablero BI integrador", fillcolor="#123b5d", fontcolor="white"];
 s [label="Escenarios prescriptivos", fillcolor="#FFF1E8", color="#cf6225"];
 d->f; d->e; f->i [label=" demanda y error"]; e->i [label=" ε"]; f->bi; e->bi; i->bi; bi->s; }
""", width="stretch")

    st.markdown("#### Objetivos específicos, indicadores, meta y resultado")
    st.caption("La columna Resultado se calcula en vivo con los modelos y artefactos de la app.")
    w_long = long.holdout_WAPE
    beat = int((reg.set_index("sku").holdout_WAPE <= naive_hold.reindex(reg.sku).values).sum())
    sig = int((el.p_cluster < .05).sum())
    rows = [
        {"Dimensión": "1 · Forecast",
         "Objetivo específico": "Pronosticar la demanda semanal por SKU eligiendo un modelo campeón con validación temporal (calibración 4 sem., selección H12 y comprobación H12 independientes).",
         "Indicador": "WAPE (%) y MAE de comprobación",
         "Meta": "WAPE de comprobación ≤ 15% en los SKU con ≥ 104 semanas de historia",
         "Resultado": f"{int((w_long <= 15).sum())}/{len(long)} SKU cumplen (mediana {w_long.median():.1f}%); el campeón iguala o mejora al Naive en comprobación en {beat}/{len(reg)} SKU"},
        {"Dimensión": "2 · Elasticidad",
         "Objetivo específico": "Estimar la elasticidad precio-demanda por SKU con regresión log-log y controles, con IC 95% agrupado por semana, y evaluar su significancia.",
         "Indicador": "β ln(precio), p-value agrupado, IC 95%, R²",
         "Meta": "30/30 SKU estimados con IC 95%; significancia evaluada al 5%",
         "Resultado": f"{int(el.ci_low.notna().sum())}/30 SKU con IC; R² entre {el.r2.min():.2f} y {el.r2.max():.2f}; {sig} SKU significativos al 5%"},
        {"Dimensión": "3 · Inventarios",
         "Objetivo específico": "Traducir el forecast y su error fuera de muestra en stock de seguridad, stock objetivo y pedido sugerido (revisión periódica).",
         "Indicador": "SS, stock objetivo, servicio de ciclo simulado",
         "Meta": "Servicio de ciclo simulado ≥ 95% en 30/30 SKU",
         "Resultado": f"{int((inv.servicio_simulado >= .95).sum())}/30 SKU; SS promedio {inv.SS.mean():,.0f} u."},
        {"Dimensión": "4 · Integración BI",
         "Objetivo específico": "Integrar forecast, elasticidad e inventario en un tablero filtrable por SKU, modelo y periodo, y evaluar escenarios.",
         "Indicador": "SKU integrados y escenarios comparados",
         "Meta": "100% de SKU con las 3 métricas en una vista ejecutiva y ≥ 3 escenarios comparados",
         "Resultado": "30/30 SKU integrados; 4 vistas BI + comparación de escenarios de precio y servicio"},
    ]
    st.table(pd.DataFrame(rows).set_index("Dimensión"))
    st.caption("Fuente de datos: FMCG daily sales data 2022-2024 (Kaggle), agregada a frecuencia semanal.")


# ============================================================
# Página: Datos y análisis descriptivo
# ============================================================
def page_datos():
    st.subheader("Datos y análisis descriptivo")
    df = raw()
    sec = st.segmented_control("Sección", ["Estadística descriptiva", "Correlación", "Estandarización",
                                           "Reducción de dimensiones (PCA)"],
                               default="Estadística descriptiva", label_visibility="collapsed") or "Estadística descriptiva"
    if sec == "Estadística descriptiva":
        a, b, c, d = st.columns(4)
        a.metric("Registros", f"{len(df):,}".replace(",", " "))
        b.metric("SKU", df.sku.nunique())
        c.metric("Canales × regiones", f"{df.channel.nunique()} × {df.region.nunique()}")
        d.metric("Periodo", f"{df.week.min():%d/%m/%Y} – {df.week.max():%d/%m/%Y}")
        scope = st.radio("Alcance", ["Base completa", "10 SKU priorizados", "SKU de la barra lateral"], horizontal=True)
        x = df
        if scope == "10 SKU priorizados":
            x = df[df.sku.isin(priority_skus())]
        elif scope == "SKU de la barra lateral":
            x = df[df.sku == st.session_state.get("sku")]
        t = describe(x, NUMERIC_VARS)
        st.dataframe(t.style.format({k: "{:,.3f}" for k in t.columns if k not in ("Variable", "Descripción", "n")} | {"n": "{:,.0f}"}),
                     hide_index=True, width="stretch")
        st.download_button("Descargar estadística descriptiva (CSV)", t.to_csv(index=False).encode("utf-8"),
                           "estadistica_descriptiva.csv", "text/csv")
        cat = []
        for c_ in ["sku", "channel", "region", "lifecycle_stage"]:
            vc = x[c_].value_counts()
            cat.append({"Variable": c_, "Categorías": x[c_].nunique(), "Moda": vc.index[0], "Frecuencia de la moda": int(vc.iloc[0])})
        st.dataframe(pd.DataFrame(cat), hide_index=True, width="stretch")
        var = st.selectbox("Distribución de la variable", NUMERIC_VARS, format_func=lambda v: f"{VAR_LABELS[v]} ({v})")
        s = pd.to_numeric(x[var], errors="coerce").dropna()
        c1, c2 = st.columns([1.6, 1])
        f = px.histogram(s, nbins=40, color_discrete_sequence=[BASE], title=f"Histograma · {VAR_LABELS[var]}")
        vline(f, s.mean(), f"Media {s.mean():.2f}", "dash")
        f.update_layout(**LAYOUT, showlegend=False, height=380)
        with c1:
            plot(f)
        f = px.box(s, points="outliers", color_discrete_sequence=[BASE], title="Diagrama de caja")
        f.update_layout(**LAYOUT, showlegend=False, height=380)
        with c2:
            plot(f)
        st.caption("Se excluyen 5 columnas «Unnamed» vacías; existe un valor no numérico (`q`) en units_sold de MI-006 "
                   "(primera semana) que se trata como faltante.")
    elif sec == "Correlación":
        cols = ["units_sold", "price_unit", "promotion_flag", "stock_available", "delivery_days", "is_holiday_week",
                "is_summer", "is_winter", "lag_1", "rolling_mean_4", "rolling_std_4", "momentum"]
        m = st.radio("Coeficiente", ["pearson", "spearman"], horizontal=True)
        corr = df[cols].corr(method=m)
        f = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                      title="Matriz de correlación")
        f.update_layout(**{**LAYOUT, "legend": {}}, height=620)
        plot(f)
        st.caption("Diagnóstico descriptivo y de redundancia (p. ej. lag_1 ↔ rolling_mean_4); no implica causalidad.")
    elif sec == "Estandarización":
        st.markdown("Antes de Ridge, Lasso/regresiones penalizadas y PCA se aplica la estandarización z-score "
                    "(`StandardScaler`): **z = (x − μ) / σ**, que deja cada variable con media 0 y desviación 1.")
        cols = ["units_sold", "price_unit", "promotion_flag", "stock_available", "delivery_days", "rolling_mean_4"]
        X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        Z = (X - X.mean()) / X.std(ddof=0)
        t = pd.DataFrame({"Variable": cols, "Media original": X.mean().values, "Desv. original": X.std(ddof=0).values,
                          "Mín.": X.min().values, "Máx.": X.max().values, "Media z": Z.mean().round(10).values,
                          "Desv. z": Z.std(ddof=0).values, "Mín. z": Z.min().values, "Máx. z": Z.max().values})
        st.dataframe(t.style.format({k: "{:,.3f}" for k in t.columns if k != "Variable"}), hide_index=True, width="stretch")
        v = st.selectbox("Comparar distribución", cols, index=1)
        c1, c2 = st.columns(2)
        for col_, data_, title, color in [(c1, X[v], "escala original", "#94A3B8"), (c2, Z[v], "estandarizada (z)", BASE)]:
            f = px.histogram(data_, nbins=40, title=f"{v} · {title}", color_discrete_sequence=[color])
            f.update_layout(**LAYOUT, showlegend=False, height=320)
            with col_:
                plot(f)
        st.info("Se usa en Fourier + Ridge (StandardScaler dentro del pipeline), en la eliminación de columnas redundantes "
                "del OLS (rango con variables escaladas) y en el PCA.")
    else:
        st.markdown("Matriz **SKU × 10 indicadores** (ingresos, unidades, precio promedio, participación promocional, stock "
                    "promedio, días de entrega, WAPE de comprobación del campeón, elasticidad OLS, stock de seguridad y "
                    "stock objetivo). Se estandariza y se aplica PCA reteniendo el mínimo de componentes que explique el umbral.")
        c1, c2 = st.columns(2)
        thr = c1.slider("Umbral de varianza acumulada", .6, .95, .8, .05)
        group = c2.radio("SKU", ["30 SKU", "10 SKU priorizados"], horizontal=True)
        summ = fc.sku_summary(df).set_index("sku")
        el = elasticity().set_index("sku")
        inv = default_inventory().set_index("sku")
        reg = registry().set_index("sku")
        skus = reg.index.tolist() if group == "30 SKU" else priority_skus()
        Mx = pd.DataFrame([{
            "sku": s, "Ingresos": summ.loc[s, "total_revenue"], "Unidades": summ.loc[s, "total_units"],
            "Precio promedio": df.loc[df.sku == s, "price_unit"].mean(),
            "Participación promocional": df.loc[df.sku == s, "promotion_flag"].mean(),
            "Stock promedio": df.loc[df.sku == s, "stock_available"].mean(),
            "Días de entrega": df.loc[df.sku == s, "delivery_days"].mean(),
            "WAPE campeón": reg.loc[s, "holdout_WAPE"], "Elasticidad": el.loc[s, "elasticity"],
            "Stock de seguridad": inv.loc[s, "SS"], "Stock objetivo": inv.loc[s, "S"]} for s in skus]).set_index("sku")
        Z = StandardScaler().fit_transform(Mx.values)
        pca = PCA().fit(Z)
        evr, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
        k = int(np.argmax(cum >= thr) + 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Variables originales", Mx.shape[1])
        m2.metric("Componentes retenidos", k)
        m3.metric("Varianza explicada", f"{100 * cum[k - 1]:.2f}%")
        pcs = [f"PC{i + 1}" for i in range(len(evr))]
        f = go.Figure()
        f.add_bar(x=pcs, y=100 * evr, name="Individual", marker_color=INK)
        f.add_scatter(x=pcs, y=100 * cum, name="Acumulada", mode="lines+markers", line=dict(color=SCEN))
        f.add_hline(y=100 * thr, line_dash="dash")
        f.update_layout(**LAYOUT, height=380, yaxis_title="% varianza", title="Varianza explicada por componente")
        plot(f)
        lo = pd.DataFrame(pca.components_[:k].T, index=Mx.columns, columns=pcs[:k])
        sc = pd.DataFrame(pca.transform(Z)[:, :max(k, 2)], index=Mx.index, columns=pcs[:max(k, 2)]).reset_index()
        c1, c2 = st.columns([1, 1.2])
        f = px.imshow(lo, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Cargas")
        f.update_layout(**{**LAYOUT, "legend": {}}, height=440)
        with c1:
            plot(f)
        f = px.scatter(sc, x="PC1", y="PC2", text="sku", title="SKU en PC1–PC2")
        f.update_traces(marker=dict(size=11, color=INK), textposition="top center")
        f.update_layout(**LAYOUT, height=440)
        with c2:
            plot(f)
        st.caption("PCA es exploratorio: resume el perfil de los SKU; no reemplaza los modelos de cada dimensión.")


# ============================================================
# Página: Tablero BI integrador (réplica del PBIX)
# ============================================================
def page_bi():
    st.subheader("Tablero BI integrador")
    st.caption("Réplica web de las 4 páginas del reporte Power BI (forecast, inventarios, elasticidad y mapa ejecutivo). "
               "En Power BI cada visual R recibe el dataset filtrado por los segmentadores; aquí los filtros cumplen ese rol.")
    reg = registry()
    all_skus = reg.sku.sort_values().tolist()
    top = priority_skus()
    grp = st.radio("Conjunto de SKU", ["10 SKU priorizados (como el PBIX)", "30 SKU"], horizontal=True, key="bi_grp")
    pool = top if grp.startswith("10") else all_skus
    t1, t2, t3, t4, t5 = st.tabs(["Forecast", "Inventarios", "Elasticidad", "Mapa ejecutivo", "Descargas"])

    with t1:
        fl = all_models_forecast()
        c1, c2 = st.columns([2, 1])
        sel = c1.multiselect("Segmentador · SKU", pool, default=[], placeholder="Todos los SKU del conjunto", key="bi_f_sku") or pool
        models = c2.multiselect("Segmentador · Modelo", NAMES, default=NAMES, key="bi_f_model") or NAMES
        single = len(sel) == 1
        f = fl[fl.sku.isin(sel) & fl.model.isin(models)].sort_values(["sku", "model", "week"]).copy()
        f["t"] = f.groupby(["sku", "model"]).cumcount() + 1
        parts = []
        for s in sel:
            h = history(s)[["week", "units_sold"]].tail(79).copy()
            h["t"] = np.arange(-len(h) + 1, 1)
            parts.append(h)
        h = pd.concat(parts)
        xcol = "week" if single else "t"
        act = h.groupby(xcol)["units_sold"].sum().reset_index()
        agg = f.groupby([xcol, "model"])["value"].sum().reset_index()
        lx, lv = act[xcol].iloc[-1], act["units_sold"].iloc[-1]
        left, right = st.columns([1.05, 1])
        with left:
            fig = go.Figure()
            fig.add_scatter(x=act[xcol], y=act.units_sold, name="Realidad", line=dict(color=MODEL_COLORS["Realidad"]))
            dashes = {fc.MODEL_NAIVE: "dash", fc.MODEL_RIDGE: "solid", fc.MODEL_SARIMAX: "solid", fc.MODEL_DHR: "dashdot"}
            for m in models:
                d = agg[agg.model == m].sort_values(xcol)
                fig.add_scatter(x=[lx, *d[xcol]], y=[lv, *d.value], name=m, line=dict(color=MODEL_COLORS[m], dash=dashes[m], width=2.5))
            if single:
                vline(fig, lx, "Inicio forecast")
            else:
                fig.add_vline(x=0, line_dash="dot", line_color="#64748B")
            fig.update_layout(**LAYOUT, height=480, yaxis_title="Unidades vendidas",
                              title=f"Comparativa de modelos · {sel[0] if single else f'{len(sel)} SKU (consolidado)'}",
                              xaxis_title="Semana" if single else "Semanas relativas al inicio del forecast")
            plot(fig)
            if not single:
                st.caption("Con varios SKU se suma por semana y modelo, alineando cada SKU por su última semana observada.")
        with right:
            pr = f.groupby(["sku", "model"])["value"].sum().reset_index()
            order = pr.groupby("sku")["value"].mean().sort_values().index.tolist()
            win = reg.set_index("sku").winner
            pr["★"] = np.where(pr.apply(lambda r: win[r.sku] == r.model, axis=1), "★", "")
            fig = px.bar(pr, x="value", y="sku", color="model", barmode="group", orientation="h", text="★",
                         color_discrete_map=MODEL_COLORS, category_orders={"sku": order, "model": NAMES},
                         labels={"value": "Unidades proyectadas (12 semanas)", "sku": "SKU", "model": "Modelo"},
                         title="Demanda proyectada por SKU (★ = campeón)")
            fig.update_traces(textposition="outside")
            fig.update_layout(**LAYOUT, height=max(480, 26 * len(sel) + 200))
            plot(fig)
        mt = metrics()
        sc = st.radio("WAPE de", ["holdout", "selection"], horizontal=True, key="bi_f_scope",
                      format_func=lambda s: "Comprobación (holdout)" if s == "holdout" else "Selección")
        pv = mt[mt.scope.eq(sc) & mt.sku.isin(sel) & mt.model.isin(models)].pivot(index="sku", columns="model", values="WAPE")
        st.dataframe(pv.style.format("{:.2f}").highlight_min(axis=1, color="#C8E6C9"), width="stretch")

    with t2:
        rows = bi_inventory_rows()
        c1, c2 = st.columns([1, 2])
        years = sorted(rows.year.dropna().unique().astype(int))
        ys = c1.multiselect("Segmentador · Año", years, default=[], placeholder="Todos los años", key="bi_i_year")
        only = c2.toggle("Solo el SKU de la barra lateral", key="bi_i_one")
        d = rows[rows.sku.isin(pool)]
        if ys:
            d = d[d.year.isin(ys)]
        if only:
            d = rows[rows.sku == st.session_state.get("sku")]
            if ys:
                d = d[d.year.isin(ys)]
        st.caption("Parámetros del script R del PBIX: servicio 95% · L = delivery_days/7 · σL = 0.1 sem. · R = 1 sem. · "
                   "D = rolling_mean_4 · σD = rolling_std_4, por fila canal-región-semana.")
        left, right = st.columns([1.1, 1])
        with left:
            w = d.groupby("week").agg(stock=("stock_available", "mean"), rop=("Punto_Reorden_ROP", "mean"),
                                      ss=("Stock_Seguridad", "mean"), smax=("Inventario_Maximo_S", "mean")).reset_index()
            al = w[w.stock < w.rop]
            fig = go.Figure()
            fig.add_scatter(x=w.week, y=w.ss, fill="tozeroy", fillcolor="rgba(192,57,43,.15)", line=dict(width=0), name="Zona crítica", hoverinfo="skip")
            fig.add_scatter(x=w.week, y=w.smax, name="Inventario máximo (S)", line=dict(color="#2B5C8F", dash="dash"))
            fig.add_scatter(x=w.week, y=w.rop, name="Punto de reorden (ROP)", line=dict(color="#E67E22", width=2))
            fig.add_scatter(x=w.week, y=w.ss, name="Stock de seguridad (SS)", line=dict(color="#C0392B", dash="dot"))
            fig.add_scatter(x=w.week, y=w.stock, name="Stock disponible real", line=dict(color="#27AE60", width=2))
            if not al.empty:
                fig.add_scatter(x=al.week, y=al.stock, mode="markers", name="Alerta: stock < ROP", marker=dict(color="#D9534F", size=8))
            fig.update_layout(**LAYOUT, height=470, hovermode="x unified", yaxis_title="Unidades (promedio por fila)",
                              title="Control de inventario y alertas de reabastecimiento")
            plot(fig)
            st.caption(f"Semanas con alerta (stock < ROP): {len(al)} de {len(w)}.")
        with right:
            last = d.week.max()
            s = (d[d.week == last].groupby("sku").agg(stock=("stock_available", "sum"), rop=("Punto_Reorden_ROP", "sum"),
                                                       ss=("Stock_Seguridad", "sum"), smax=("Inventario_Maximo_S", "sum")).reset_index())
            s["Brecha"] = s.stock - s.rop
            s["Estado"] = [classify(*v) for v in zip(s.stock, s.ss, s.rop, s.smax)]
            fig = px.bar(s.sort_values("Brecha"), x="Brecha", y="sku", color="Estado", orientation="h",
                         color_discrete_map=STATUS_COLORS, category_orders={"Estado": STATUS_ORDER},
                         labels={"Brecha": "Unidades disponibles − ROP", "sku": "SKU"},
                         title=f"Estado operativo por SKU (semana {last:%d/%m/%Y})")
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(**LAYOUT, height=max(470, 22 * len(s) + 220))
            plot(fig)

    with t3:
        el = elasticity()
        el = el[el.sku.isin(pool)].copy()
        c1, c2 = st.columns([1, 3])
        var = c1.select_slider("Parámetro · Variación de precio", list(range(-20, 21, 5)), value=0,
                               format_func=lambda v: f"{v:+d}%", key="bi_e_var")
        base_e = c2.radio("Elasticidad para proyectar", ["OLS (como el PBIX)", "Sensibilidad del escenario (app)"],
                          horizontal=True, key="bi_e_src")
        e_used = el.elasticity if base_e.startswith("OLS") else el.elasticity_used
        el["Unidades proyectadas"] = el.unidades_totales * (1 + var / 100) ** e_used
        el["Δ unidades"] = el["Unidades proyectadas"] - el.unidades_totales
        left, right = st.columns([1, 1.1])
        with left:
            e2 = el.sort_values("elasticity")
            fig = go.Figure()
            for sig, col in [("Sí", "#1E40AF"), ("No", "#94A3B8")]:
                q = e2[e2.significativo_5pct == sig]
                fig.add_scatter(x=q.elasticity, y=q.sku, mode="markers", name=f"Significativo: {sig}", marker=dict(size=11, color=col),
                                error_x=dict(type="data", symmetric=False, array=q.ci_high - q.elasticity,
                                             arrayminus=q.elasticity - q.ci_low, color=col))
            fig.add_vline(x=0, line_dash="dash", line_color="#64748B")
            fig.update_layout(**LAYOUT, height=max(480, 20 * len(e2) + 200), xaxis_title="Elasticidad OLS (IC 95% agrupado)",
                              title="Elasticidad precio de la demanda por SKU",
                              yaxis=dict(categoryorder="array", categoryarray=e2.sku.tolist()))
            plot(fig)
        with right:
            fig = px.scatter(el, x="elasticity", y="unidades_totales", size="precio_promedio", color="significativo_5pct",
                             text="sku", size_max=24, color_discrete_map={"Sí": "#1E40AF", "No": "#94A3B8"},
                             labels={"elasticity": "Elasticidad", "unidades_totales": "Unidades totales", "significativo_5pct": "Significativo"},
                             title="Volumen vs. elasticidad (tamaño = precio promedio)")
            fig.update_traces(textposition="top center")
            fig.update_layout(**LAYOUT, height=320)
            plot(fig)
            t = el[["sku", "unidades_totales", "precio_promedio", "elasticity", "p_cluster", "elasticity_used",
                    "clasificacion", "Unidades proyectadas", "Δ unidades"]]
            t.columns = ["SKU", "Unidades", "Precio prom.", "Elasticidad OLS", "p-value", "ε escenario", "Clasificación",
                         "Unidades proyectadas", "Δ unidades"]
            st.dataframe(t.style.format({"Unidades": "{:,.0f}", "Precio prom.": "{:.2f}", "Elasticidad OLS": "{:+.4f}",
                                         "p-value": "{:.3f}", "ε escenario": "{:+.4f}", "Unidades proyectadas": "{:,.0f}",
                                         "Δ unidades": "{:+,.0f}"}), hide_index=True, width="stretch", height=260)
        st.caption("Ningún SKU tiene una elasticidad OLS significativa al 5% con errores agrupados por semana. La "
                   "«sensibilidad del escenario» es la que usa el simulador de la app (condicionada a ε ≤ 0).")

    with t4:
        inv = default_inventory()
        fl = all_models_forecast()
        wf = fl[fl.is_winner]
        st_ = bi_status_latest()
        el = elasticity()
        c1, c2 = st.columns(2)
        yrs = sorted(wf.week.dt.year.unique())
        ys = c1.multiselect("Segmentador · Año del forecast", yrs, default=[], placeholder="Todo el horizonte", key="bi_x_y")
        ms = c2.multiselect("Segmentador · Modelo ganador", NAMES, default=[], placeholder="Todos", key="bi_x_m")
        w = wf if not ys else wf[wf.week.dt.year.isin(ys)]
        ex = (w.groupby("sku").value.sum().rename("forecast_12w").reset_index()
              .merge(el[["sku", "elasticity", "p_cluster", "elasticity_used"]], on="sku")
              .merge(inv[["sku", "modelo_campeon", "SS", "ROP", "S"]], on="sku")
              .merge(st_[["sku", "stock", "estado"]], on="sku")
              .merge(registry()[["sku", "holdout_WAPE"]], on="sku"))
        ex = ex[ex.sku.isin(pool)]
        if ms:
            ex = ex[ex.modelo_campeon.isin(ms)]
        if ex.empty:
            st.warning("No hay SKU con esa combinación de filtros.")
        else:
            k = st.columns(4)
            k[0].metric("Demanda proyectada", f"{ex.forecast_12w.sum():,.0f} u.")
            k[1].metric("WAPE prom. del campeón", f"{ex.holdout_WAPE.mean():.2f}%")
            k[2].metric("SKU elasticidad sig. (5%)", int((ex.p_cluster < .05).sum()))
            k[3].metric("SKU a reordenar", int(ex.estado.isin(["Agotamiento crítico", "Reordenar urgente"]).sum()))
            fig = px.scatter(ex, x="elasticity", y="forecast_12w", size="ROP", color="estado", text="sku", size_max=40,
                             color_discrete_map=STATUS_COLORS, category_orders={"estado": STATUS_ORDER},
                             hover_data={"modelo_campeon": True, "holdout_WAPE": ":.2f", "p_cluster": ":.3f"},
                             labels={"elasticity": "Elasticidad precio-demanda (OLS)", "forecast_12w": "Demanda proyectada acumulada",
                                     "estado": "Estado de inventario"},
                             title="Mapa integrado: demanda, inventario y precio (tamaño = ROP)")
            fig.add_vline(x=0, line_width=1)
            fig.update_traces(textposition="top center")
            fig.update_layout(**LAYOUT, height=520)
            plot(fig)
            ex["Acción sugerida"] = np.select(
                [ex.estado.isin(["Agotamiento crítico", "Reordenar urgente"]), ex.estado.eq("Sobre-stock")],
                ["Emitir orden de reposición", "Frenar compras / evaluar promoción"], default="Mantener política")
            show = ex[["sku", "modelo_campeon", "holdout_WAPE", "forecast_12w", "elasticity", "p_cluster", "elasticity_used",
                       "SS", "ROP", "S", "estado", "Acción sugerida"]]
            show.columns = ["SKU", "Modelo campeón", "WAPE %", "Forecast 12 sem.", "Elasticidad OLS", "p-value",
                            "ε escenario", "SS", "ROP", "Stock objetivo", "Estado (PBIX)", "Acción sugerida"]
            st.dataframe(show.style.format({"WAPE %": "{:.2f}", "Forecast 12 sem.": "{:,.0f}", "Elasticidad OLS": "{:+.3f}",
                                            "p-value": "{:.3f}", "ε escenario": "{:+.4f}", "SS": "{:,.0f}", "ROP": "{:,.0f}",
                                            "Stock objetivo": "{:,.0f}"}), hide_index=True, width="stretch")
            st.caption("Tamaño (ROP), SS y stock objetivo: motor de la app con la política académica por defecto (L = 2, "
                       "σL = 0.5, R = 1, servicio 95%). Estado: lógica del script R del PBIX con el stock_available de la "
                       "última semana. Los stocks del Excel son datos del curso, no inventario empresarial verificado.")

    with t5:
        st.write("Tablas que alimentan este tablero (equivalentes a las tablas BI_* del PBIX).")
        hist_all = pd.concat([history(s)[["week", "units_sold"]].assign(sku=s) for s in all_skus])
        bi_fc = pd.concat([hist_all.rename(columns={"units_sold": "value"}).assign(record_type="Actual", model="Actual", is_winner=False),
                           all_models_forecast().assign(record_type="Forecast")])[["sku", "week", "record_type", "model", "value", "is_winner"]]
        files = {"BI_Forecast.csv": bi_fc, "BI_Metrics.csv": metrics(), "BI_Elasticidad.csv": elasticity(),
                 "BI_Inventario.csv": default_inventory(), "BI_Estado_PBIX.csv": bi_status_latest()}
        cols = st.columns(3)
        for i, (n, t) in enumerate(files.items()):
            cols[i % 3].download_button(n, t.to_csv(index=False).encode("utf-8-sig"), n, "text/csv", width="stretch")


# ============================================================
# Página: Comparar escenarios (prescriptivo)
# ============================================================
def page_escenarios(config, run):
    st.subheader("Comparación prescriptiva de escenarios")
    st.write("Cada fila es un escenario completo del motor (forecast campeón → respuesta al precio → simulación de "
             "reposición). Se usan los mismos supuestos de stock, plazos y elasticidad definidos en la barra lateral.")
    c1, c2 = st.columns([2, 1])
    with c1:
        default = pd.DataFrame({"Escenario": ["A · Descuento", "B · Base", "C · Alza de precio"],
                                "Cambio de precio (%)": [-10., 0., 10.], "Servicio de ciclo (%)": [97.5, 95., 95.]})
        tab = st.data_editor(default, hide_index=True, num_rows="dynamic", key="cmp_editor", column_config={
            "Cambio de precio (%)": st.column_config.NumberColumn(min_value=-40., max_value=40., step=.5),
            "Servicio de ciclo (%)": st.column_config.NumberColumn(min_value=80., max_value=99., step=.5)})
    scope = c2.radio("Alcance", ["SKU de la barra lateral", "10 SKU priorizados"], key="cmp_scope")
    c2.caption("Para el conjunto se usan los supuestos académicos por defecto de cada SKU (stock en cero salvo que lo "
               "hayas editado en la sesión).")
    tab = tab.dropna()
    tab = tab[tab.Escenario.astype(str).str.strip() != ""].drop_duplicates("Escenario")
    if tab.empty:
        st.warning("Agrega al menos un escenario.")
        return
    from dataclasses import asdict, replace
    skus = [config.sku] if scope.startswith("SKU") else priority_skus()
    rows = []
    with st.spinner("Simulando escenarios…"):
        for _, sc in tab.iterrows():
            for sku in skus:
                basecfg = config if sku == config.sku else M.Scenario(**st.session_state.profiles.get(sku, asdict(M.Scenario(sku=sku))))
                cfg = replace(basecfg, price_change_pct=float(sc["Cambio de precio (%)"]), service=float(sc["Servicio de ciclo (%)"]) / 100)
                r = run(cfg)
                iv = r["inventory"].set_index("scenario").loc["Precio propuesto"]
                s = r["summary"]
                rows.append({"Escenario": sc.Escenario, "sku": sku, "precio": s["price_proposed"], "unidades": s["scenario_units"],
                             "ingreso": s["scenario_revenue"], "SS": float(iv.safety_stock), "objetivo": float(iv.target_stock),
                             "pedido": float(iv.suggested_order), "servicio_sim": float(iv.simulated_cycle_service),
                             "inv_prom": float(iv.safety_stock) + s["scenario_units"] / max(cfg.horizon, 1) * cfg.review_weeks / 2})
    res = pd.DataFrame(rows)
    agg = res.groupby("Escenario", sort=False).agg(unidades=("unidades", "sum"), ingreso=("ingreso", "sum"), SS=("SS", "sum"),
                                                   objetivo=("objetivo", "sum"), pedido=("pedido", "sum"),
                                                   servicio=("servicio_sim", "mean"), inv_prom=("inv_prom", "sum")).reset_index()
    agg = agg.merge(tab.rename(columns={"Cambio de precio (%)": "dp", "Servicio de ciclo (%)": "ns"}), on="Escenario")
    ref = agg.loc[agg.dp.abs().idxmin()]
    agg["Δ ingreso %"] = 100 * (agg.ingreso / ref.ingreso - 1)
    agg["Ingreso / u. inventario"] = agg.ingreso / agg.inv_prom
    cols = st.columns(min(len(agg), 4))
    for i, r in agg.iterrows():
        with cols[i % len(cols)]:
            st.metric(r.Escenario, f"{r.ingreso:,.0f} u.m.", f"{r['Δ ingreso %']:+.2f}% vs {ref.Escenario}" if r.Escenario != ref.Escenario else "referencia")
            st.caption(f"Precio {r.dp:+.1f}% · servicio {r.ns:.1f}% · {r.unidades:,.0f} u. · SS {r.SS:,.0f} u. · pedido {r.pedido:,.0f} u.")
    fig = go.Figure()
    fig.add_bar(x=agg.Escenario, y=agg.ingreso, name="Ingreso potencial (u.m.)", marker_color=INK)
    fig.add_scatter(x=agg.Escenario, y=agg.SS, name="Stock de seguridad (u.)", yaxis="y2", mode="lines+markers",
                    line=dict(color=SCEN, width=3), marker=dict(size=12))
    fig.update_layout(**LAYOUT, height=400, title="Ingreso esperado vs. inventario de protección",
                      yaxis=dict(title="Ingreso (u.m.)"), yaxis2=dict(title="SS (u.)", overlaying="y", side="right"))
    plot(fig)
    t = agg[["Escenario", "dp", "ns", "unidades", "ingreso", "Δ ingreso %", "SS", "objetivo", "pedido", "servicio", "Ingreso / u. inventario"]]
    t.columns = ["Escenario", "Precio %", "Servicio %", "Unidades", "Ingreso", "Δ ingreso %", "SS", "Stock objetivo",
                 "Pedido sugerido", "Servicio simulado", "Ingreso / u. inventario"]
    st.dataframe(t.style.format({"Precio %": "{:+.1f}", "Servicio %": "{:.1f}", "Unidades": "{:,.0f}", "Ingreso": "{:,.0f}",
                                 "Δ ingreso %": "{:+.2f}", "SS": "{:,.0f}", "Stock objetivo": "{:,.0f}", "Pedido sugerido": "{:,.0f}",
                                 "Servicio simulado": "{:.1%}", "Ingreso / u. inventario": "{:,.2f}"}), hide_index=True, width="stretch")
    ok = agg[agg.servicio >= agg.ns / 100 - 0.005]
    best = (ok if not ok.empty else agg).sort_values(["Ingreso / u. inventario", "ingreso"], ascending=False).iloc[0]
    st.success(f"**Escenario sugerido: {best.Escenario}** (precio {best.dp:+.1f}%, servicio {best.ns:.1f}%). Es el que obtiene "
               f"más ingreso por unidad de inventario mantenido ({best['Ingreso / u. inventario']:,.2f}) entre los que cumplen "
               "su servicio de ciclo en la simulación.")
    st.caption("Criterio: 1) cumplir el servicio de ciclo objetivo en la simulación; 2) mayor ingreso por unidad de "
               "inventario promedio (SS + D·R/2). Sin costos ni márgenes reales no se calcula utilidad; la respuesta al "
               "precio es una sensibilidad condicionada, no un efecto causal demostrado.")
    with st.expander("Detalle por SKU y escenario"):
        st.dataframe(res, hide_index=True, width="stretch")


# ============================================================
# Fundamentación (se agrega a «Método y fuentes»)
# ============================================================
REFERENCES = [
    "Burnay, C., Lega, M., & Bouraga, S. (2024). Business intelligence and cognitive loads: Proposition of a dashboard adoption model. *Data & Knowledge Engineering, 152*, 102310. https://doi.org/10.1016/j.datak.2024.102310",
    "de Castro Moraes, T., Yuan, X.-M., & Chew, E. P. (2024). Hybrid convolutional long short-term memory models for sales forecasting in retail. *Journal of Forecasting, 43*(5), 1278–1293. https://doi.org/10.1002/for.3073",
    "Holmes, A., Illowsky, B., & Dean, S. (2023). *Introductory business statistics* (2nd ed.). OpenStax.",
    "Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/",
    "Jeon, Y., Hoang, H., Thompson, W., & Abler, D. (2024). A meta-analysis of U.S. food demand elasticities to detect the impacts of scanner data. *Applied Economic Perspectives and Policy, 46*(2), 760–780. https://doi.org/10.1002/aepp.13414",
    "Kačmáry, P., Bindzár, P., Kovalčík, J., & Ondov, M. (2024). Forecast of sales of selected food products in retail using Fourier series analysis and non-linear regression. *Foresight, 26*(3), 487–504. https://doi.org/10.1108/FS-12-2022-0168",
    "Keller, K. O., & Sethuraman, R. (2026). Promotion elasticities of national and store brands. *Journal of Retailing, 102*(3), 717–734. https://doi.org/10.1016/j.jretai.2026.01.001",
    "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting, 38*(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013",
    "Mudau, T. N., Cohen, J., & Papageorgiou, E. (2026). Tracing the effects of business intelligence on management controls and organisational performance. *Journal of Enterprise Information Management*, 1–29. https://doi.org/10.1108/JEIM-10-2024-0581",
    "Oesterreich, T. D., Anton, E., & Teuteberg, F. (2022). What translates big data into business value? *Information & Management, 59*(6), 103685. https://doi.org/10.1016/j.im.2022.103685",
    "Shi, J. (2022). Application of the model combining demand forecasting and inventory decision in feature based newsvendor problem. *Computers & Industrial Engineering, 173*, 108709. https://doi.org/10.1016/j.cie.2022.108709",
    "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B, 58*(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",
    "Trapero, J. R., Holgado de Frutos, E., & Pedregal, D. J. (2024). Demand forecasting under lost sales stock policies. *International Journal of Forecasting, 40*(3), 1055–1068. https://doi.org/10.1016/j.ijforecast.2023.09.004",
    "Wolters, J., & Huchzermeier, A. (2021). Joint in-season and out-of-season promotion demand forecasting in a retail environment. *Journal of Retailing, 97*(4), 726–745. https://doi.org/10.1016/j.jretai.2021.01.003",
    "Ye, L., Xie, N., Boylan, J. E., & Shang, Z. (2024). Forecasting seasonal demand for retail: A Fourier time-varying grey model. *International Journal of Forecasting, 40*(4), 1467–1485. https://doi.org/10.1016/j.ijforecast.2023.12.006",
    "Zietsman, H. J., & van Vuuren, J. H. (2023). A generic decision support framework for inventory procurement planning in distribution centres. *Computers & Industrial Engineering, 177*, 109057. https://doi.org/10.1016/j.cie.2023.109057",
]


def section_fundamentacion():
    st.subheader("Fundamentación científica de los algoritmos")
    st.markdown("""
| Dimensión | Técnica | Por qué se eligió | Evidencia |
|---|---|---|---|
| Forecast | Seasonal Naive (s = 52) | Benchmark obligatorio; en retail estacional es difícil de superar. | Makridakis et al. (2022); Hyndman & Athanasopoulos (2021) |
| Forecast | Fourier + Ridge | Estacionalidad anual con pocos armónicos (K = 1–3) + precio, promoción y calendario; la penalización L2 controla la colinealidad. | Ye et al. (2024); Wolters & Huchzermeier (2021); Kačmáry et al. (2024) |
| Forecast | SARIMAX estacional | Dependencia temporal y estacionalidad explícita a 52 semanas con exógenas. | Hyndman & Athanasopoulos (2021) |
| Forecast | DHR (Fourier + ARMA) | Modela la autocorrelación que queda en los residuos de la regresión armónica. | Hyndman & Athanasopoulos (2021); de Castro Moraes et al. (2024) |
| Forecast | Validación temporal H12 | Calibración, selección y comprobación en bloques sucesivos, sin fuga de información. | Hyndman & Athanasopoulos (2021) |
| Elasticidad | OLS log-log con controles y error agrupado | El coeficiente de ln(P) es la elasticidad; los controles separan promoción, stock y calendario. | Holmes et al. (2023); Jeon et al. (2024); Keller & Sethuraman (2026) |
| Inventarios | Revisión periódica con simulación de demanda y plazo | Conecta el error fuera de muestra del forecast con el stock de seguridad. | Trapero et al. (2024); Zietsman & van Vuuren (2023); Shi (2022) |
| Integración | Tablero BI + escenarios | El tablero reduce la carga cognitiva y facilita la decisión. | Burnay et al. (2024); Oesterreich et al. (2022); Mudau et al. (2026) |
""")
    st.subheader("¿Se usa regularización en los modelos de regresión?")
    c1, c2 = st.columns(2)
    c1.markdown("""
**Sí, en el forecast: Ridge (L2).**
`min Σ(y − Xβ)² + α·Σβ²`, con predictores estandarizados. α ∈ {0.1, 1, 10, 100} se calibra por SKU junto con
K ∈ {1, 2, 3} y tendencia lineal/cuadrática, usando solo el bloque de calibración (antes de selección y comprobación).
""")
    c2.markdown("""
**No en la elasticidad OLS.** Penalizar ln(P) sesgaría la elasticidad hacia cero y dejaría de ser interpretable.
La parsimonia se controla retirando solo columnas exactamente redundantes (prueba de rango con variables
estandarizadas), dummies con categoría de referencia y errores agrupados por semana. Lasso (Tibshirani, 1996)
es la alternativa de selección de controles cuando se necesita descartar variables.
""")
    st.subheader("Referencias")
    for r in REFERENCES:
        st.markdown(f"- {r}")
