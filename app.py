# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import unicodedata
import re
import altair as alt  # Gráficos con eje desde 0

# =========================
# Configuración
# =========================
st.set_page_config(page_title="ABC / Súper ABC (ABC×XYZ)", layout="wide")
st.title("ABC / Súper ABC (ABC×XYZ) por Familias y SKUs")

# =========================
# Utilidades
# =========================
def to_num(series: pd.Series) -> pd.Series:
    """Convierte strings con símbolos/sep.miles a números; NaN -> 0."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace(r"[^\d\-,\.]", "", regex=True)
              .str.replace(",", "", regex=False),
        errors="coerce"
    ).fillna(0)

def normalize_text(s: pd.Series) -> pd.Series:
    """Quita tildes, recorta y colapsa espacios, pasa a MAYÚSCULAS."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
    return s.str.upper()

def pick_mode(s: pd.Series) -> str:
    """Devuelve el valor más frecuente (modo) no vacío; si no hay, devuelve ''."""
    s = s.astype(str).str.strip()
    s = s[(s != "") & s.notna()]
    if s.empty:
        return ""
    return s.value_counts().idxmax()

def abc_from_values(df: pd.DataFrame, value_col: str, a_cut=80.0, b_cut=95.0) -> pd.DataFrame:
    """Ordena DESC por value_col y asigna ABC global."""
    out = df.copy()
    out = out.sort_values([value_col], ascending=[False], kind="mergesort").reset_index(drop=True)
    total = out[value_col].sum()
    if total <= 0:
        out["%Ingresos"] = 0.0
        out["%Acum"] = 0.0
        out["Clase_ABC"] = "C"
        return out
    out["%Ingresos"] = (out[value_col] / total) * 100
    out["%Acum"] = out["%Ingresos"].cumsum()

    def clas(p):
        if p <= a_cut: return "A"
        if p <= b_cut: return "B"
        return "C"

    out["Clase_ABC"] = out["%Acum"].map(clas)
    return out

def abc_within_group(df: pd.DataFrame, group_col: str, key_col: str, value_col: str, a_cut=80.0, b_cut=95.0):
    """
    ABC de key_col dentro de cada group_col.
    Retorna: group_col, key_col, Unid_Total, Ingresos_Total, %Ingresos_Fam, %Acum_Fam, Clase_ABC_SKU
    """
    g = (df.groupby([group_col, key_col], as_index=False)
           .agg(Unid_Total=("Unid_Row", "sum"),
                Ingresos_Total=(value_col, "sum")))
    parts = []
    for fam, sub in g.groupby(group_col, sort=False):
        sub2 = sub.sort_values(["Ingresos_Total", key_col], ascending=[False, True], kind="mergesort").copy()
        t = sub2["Ingresos_Total"].sum()
        if t <= 0:
            sub2["%Ingresos_Fam"] = 0.0
            sub2["%Acum_Fam"] = 0.0
            sub2["Clase_ABC_SKU"] = "C"
        else:
            sub2["%Ingresos_Fam"] = (sub2["Ingresos_Total"] / t) * 100
            sub2["%Acum_Fam"] = sub2["%Ingresos_Fam"].cumsum()
            def clas(p):
                if p <= a_cut: return "A"
                if p <= b_cut: return "B"
                return "C"
            sub2["Clase_ABC_SKU"] = sub2["%Acum_Fam"].map(clas)
        parts.append(sub2)
    return pd.concat(parts, ignore_index=True)

def xyz_from_wide(df: pd.DataFrame, id_cols: list, month_cols: list, cv_x=0.25, cv_y=0.50,
                  level_name="Entidad"):
    """
    XYZ para datos en formato ancho (columnas = meses de UNIDADES).
    - id_cols: columnas de identificación (p.ej., ["Familia_Key"] o ["Familia_Key","SKU"])
    - month_cols: columnas de meses seleccionadas (unidades)
    Retorna DF con columnas id_cols (última renombrada a 'level_name' solo para display),
    CV y Clase_XYZ.
    """
    mtx = df[id_cols + month_cols].copy()
    for c in month_cols:
        mtx[c] = to_num(mtx[c])

    agg = mtx.groupby(id_cols, as_index=False)[month_cols].sum()
    vals = agg[month_cols].values.astype(float)
    mean = vals.mean(axis=1)
    std = vals.std(axis=1, ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(mean > 0, std / mean, np.inf)

    out = agg[id_cols].copy()
    out["CV"] = cv

    def clas_xyz(c):
        if c <= cv_x: return "X"
        if c <= cv_y: return "Y"
        return "Z"

    out["Clase_XYZ"] = out["CV"].map(clas_xyz)

    # Renombrar última id a level_name para facilitar displays si se requiere
    if level_name and isinstance(level_name, str):
        out = out.rename(columns={id_cols[-1]: level_name})
    return out

def extract_item_status(text: str):
    """
    Detecta un sufijo como '(D)', '(DXF)' o '(P.P)' al final del texto.
    Devuelve uno de: 'D', 'DXF', 'P.P' o None.
    Permite espacios y mayúsc./minúsc., y acepta '(P.P)' o '(PP)'.
    """
    s = str(text).strip()
    m = re.search(r"\(\s*(P\.?P|DXF|D)\s*\)\s*$", s, flags=re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).upper().replace(".", "")
    return {"PP": "P.P", "DXF": "DXF", "D": "D"}.get(token, None)

@st.cache_data(show_spinner=False)
def listar_hojas(file_bytes: bytes):
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names

@st.cache_data(show_spinner=False)
def leer_df(file_bytes: bytes, ext: str, sheet: str | None):
    if ext in ("xlsx", "xls"):
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    try:
        return pd.read_csv(BytesIO(file_bytes), engine="pyarrow")
    except Exception:
        return pd.read_csv(BytesIO(file_bytes))

# =========================
# Carga de archivo
# =========================
archivo = st.file_uploader("Sube un Excel (.xlsx/.xls) o CSV", type=["xlsx", "xls", "csv"])
if not archivo:
    st.info("👉 Sube tu archivo para comenzar.")
    st.stop()

ext = archivo.name.split(".")[-1].lower()
file_bytes = archivo.getvalue()

if ext in ("xlsx", "xls"):
    hoja = st.selectbox("Elige la hoja", listar_hojas(file_bytes))
    df = leer_df(file_bytes, ext, hoja)
    st.caption(f"Usando hoja **{hoja}**")
else:
    df = leer_df(file_bytes, ext, None)
    st.caption("Archivo **CSV** detectado")

st.subheader("Vista previa de datos")
st.dataframe(df.head(), use_container_width=True)

# =========================
# Selección de columnas y parámetros
# =========================
with st.form("params"):
    cols = list(df.columns)

    col_sku       = st.selectbox("Columna de **Código / SKU**", cols, key="col_sku")
    col_fam_name  = st.selectbox("Columna de **Nombre de Familia** (recomendado)", ["<ninguna>"] + cols, index=0, key="col_fam_name")
    col_fam_code  = st.selectbox("Columna de **Código de Familia** (opcional)", ["<ninguna>"] + cols, index=0, key="col_fam_code")
    col_unid      = st.selectbox("Columna de **Unidades vendidas (detalle)**", cols, key="col_unid")
    col_price     = st.selectbox("Columna de **Precio/Monto unitario**", cols, key="col_price")

    st.markdown("**Filtro por estado en descripción (opcional)**")
    col_desc = st.selectbox(
        "Columna con descripción que trae (D), (DXF), (P.P) al final (ej. D.Articulo)",
        ["<ninguna>"] + cols,
        index=0
    )
    exclude_status = st.multiselect(
        "Excluir de los cálculos estos estados",
        options=["D", "DXF", "P.P"],
        help="No borra filas del archivo original. Solo las excluye de ABC/XYZ. (D: descontinuado, DXF: descontinuado por fábrica, P.P: por pedido)"
    )

    st.markdown("**Parámetros ABC**")
    a_cut = st.slider("Corte A (%)", 50, 95, 80)
    b_cut = st.slider("Corte B (%)", a_cut+1, 99, 95)

    st.markdown("**Nivel 2 — Familias a profundizar (según clase ABC de Familias)**")
    clases_n2 = st.multiselect("Clases de familias a incluir en Nivel 2", options=["A","B","C"], default=["A"])

    st.markdown("**Súper ABC (opcional)**")
    enable_super = st.checkbox("Activar Súper ABC (combinar ABC con XYZ de variabilidad mensual)")

    month_cols = []
    cv_x = 0.25
    cv_y = 0.50
    if enable_super:
        candidates = [c for c in cols if c not in {col_sku, col_fam_name, col_fam_code, col_unid, col_price} and c != "<ninguna>"]
        st.info("Selecciona las columnas de **meses** (unidades por mes) para calcular XYZ. Usa meses contiguos (p. ej., últimos 12).")
        month_cols = st.multiselect("Columnas de meses (UNIDADES)", candidates)
        cv_x = st.number_input("Umbral X (CV ≤ X)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        cv_y = st.number_input("Umbral Y (CV ≤ Y)", min_value=cv_x, max_value=1.0, value=0.50, step=0.01)

    submitted = st.form_submit_button("Calcular")

if not submitted:
    st.stop()

# =========================
# Filtro por estado en descripción (si corresponde)
# =========================
df_excluidos = pd.DataFrame()
if col_desc != "<ninguna>" and len(exclude_status) > 0:
    df["_Estado_Articulo"] = df[col_desc].apply(extract_item_status)
    mask_excl = df["_Estado_Articulo"].isin(exclude_status)
    df_excluidos = df[mask_excl].copy()
    df = df[~mask_excl].copy()
    if not df_excluidos.empty:
        st.info(
            "Se excluyeron de los cálculos: " +
            ", ".join(f"{k}: {v}" for k, v in df_excluidos["_Estado_Articulo"].value_counts().to_dict().items())
        )
else:
    df["_Estado_Articulo"] = None

# =========================
# Preparación de base con llave de Familia
# =========================
use_code = (col_fam_code != "<ninguna>")
fam_key_source = col_fam_code if use_code else (col_fam_name if col_fam_name != "<ninguna>" else None)
if fam_key_source is None:
    st.error("Debes seleccionar al menos una columna para identificar la **Familia** (Nombre o Código).")
    st.stop()

base = df[[col_sku, fam_key_source, col_unid, col_price]].copy()
base[col_sku] = base[col_sku].astype(str).str.strip()
base[fam_key_source] = base[fam_key_source].astype(str).str.strip()
base = base[base[col_sku].ne("") & base[fam_key_source].ne("")]

# Llave normalizada
base["Familia_Key"] = normalize_text(base[fam_key_source])

# Números de detalle
base["Unid_Row"]   = to_num(base[col_unid])
base["Precio_Row"] = to_num(base[col_price])
base["Ingresos_Row"] = base["Unid_Row"] * base["Precio_Row"]

# --- Conteos globales tras filtros (exclusiones y limpieza) ---
skus_total_filtrados = base[col_sku].nunique()
familias_total_filtradas = base["Familia_Key"].nunique()
st.info(
    f"SKUs después de excluir estados y limpiar: **{skus_total_filtrados:,}**  |  "
    f"Familias: **{familias_total_filtradas:,}**"
)

# ---- Meta de Familia (Código y Nombre bonitos por Key) ----
meta_source_cols = []
if col_fam_code != "<ninguna>":
    meta_source_cols.append(col_fam_code)
if col_fam_name != "<ninguna>":
    meta_source_cols.append(col_fam_name)
if fam_key_source not in meta_source_cols:
    meta_source_cols.append(fam_key_source)

fam_meta_raw = df[meta_source_cols].copy()
fam_meta_raw["Familia_Key"] = normalize_text(fam_meta_raw[fam_key_source])

if col_fam_name != "<ninguna>":
    name_source = col_fam_name
elif col_fam_code == "<ninguna>":
    name_source = fam_key_source
else:
    name_source = None

if col_fam_code != "<ninguna>" and name_source:
    fam_meta = (fam_meta_raw.groupby("Familia_Key", as_index=False)
                .agg(Familia_Cod=(col_fam_code, pick_mode),
                     Familia_Nombre=(name_source, pick_mode)))
elif col_fam_code != "<ninguna>" and not name_source:
    fam_meta = (fam_meta_raw.groupby("Familia_Key", as_index=False)
                .agg(Familia_Cod=(col_fam_code, pick_mode)))
    fam_meta["Familia_Nombre"] = ""
else:
    fam_meta = (fam_meta_raw.groupby("Familia_Key", as_index=False)
                .agg(Familia_Nombre=(name_source, pick_mode)))
    fam_meta["Familia_Cod"] = ""

# =========================
# NIVEL 1: ABC por Familias (usando Key)
# =========================
fam_agg = (base.groupby("Familia_Key", as_index=False)
                .agg(Unid_Total=("Unid_Row", "sum"),
                     Ingresos_Total=("Ingresos_Row", "sum")))

abc_fam = abc_from_values(fam_agg, value_col="Ingresos_Total", a_cut=a_cut, b_cut=b_cut)
abc_fam = abc_fam.merge(fam_meta, on="Familia_Key", how="left")

cols_lvl1 = ["Familia_Key", "Familia_Cod", "Familia_Nombre",
             "Unid_Total", "Ingresos_Total", "%Ingresos", "%Acum", "Clase_ABC"]
st.subheader("Nivel 1: ABC por **Familias** (por ventas, agrupación robusta)")
st.dataframe(abc_fam[cols_lvl1], use_container_width=True)

# Gráfico % ventas por clase (Familias) con eje desde 0
res_fam = (abc_fam.groupby("Clase_ABC", as_index=False)
                  .agg(Ingresos=("Ingresos_Total","sum")))
res_fam["% Ingresos"] = (res_fam["Ingresos"] / res_fam["Ingresos"].sum()) * 100
res_fam = res_fam.set_index("Clase_ABC").reindex(["A","B","C"]).fillna(0)
st.markdown("**Gráfico: % de ventas por clase (Familias)**")
df_fam_plot = res_fam.reset_index().rename(columns={"% Ingresos": "Pct", "Clase_ABC": "Clase"})
chart_fam = (
    alt.Chart(df_fam_plot)
    .mark_bar()
    .encode(
        x=alt.X("Clase:N", sort=["A","B","C"], title="Clase ABC"),
        y=alt.Y("Pct:Q", title="% de ventas", scale=alt.Scale(domainMin=0, domainMax=100)),
        tooltip=["Clase:N", alt.Tooltip("Pct:Q", format=".1f")]
    )
)
st.altair_chart(chart_fam, use_container_width=True)

# =========================
# NIVEL 2: ABC por SKU dentro de Familias seleccionadas
# =========================
fams_sel_keys = abc_fam.loc[abc_fam["Clase_ABC"].isin(clases_n2), "Familia_Key"].unique().tolist()
sub_base = base[base["Familia_Key"].isin(fams_sel_keys)].copy()

# --- Conteo de SKUs en familias seleccionadas para Nivel 2 ---
skus_n2 = sub_base[col_sku].nunique() if not sub_base.empty else 0
st.info(f"SKUs en familias {clases_n2}: **{skus_n2:,}**")

if sub_base.empty:
    st.warning("No hay filas en las familias seleccionadas para Nivel 2.")
    abc_sku_n2 = pd.DataFrame()
else:
    abc_sku_n2 = abc_within_group(sub_base, group_col="Familia_Key", key_col=col_sku,
                                  value_col="Ingresos_Row", a_cut=a_cut, b_cut=b_cut)
    abc_sku_n2 = abc_sku_n2.merge(fam_meta, on="Familia_Key", how="left")
    abc_sku_n2 = abc_sku_n2.rename(columns={col_sku: "SKU"})
    st.subheader(f"Nivel 2: ABC por **SKU** dentro de Familias {clases_n2}")
    st.dataframe(
        abc_sku_n2[["Familia_Key","Familia_Cod","Familia_Nombre","SKU",
                    "Unid_Total","Ingresos_Total","%Ingresos_Fam","%Acum_Fam","Clase_ABC_SKU"]],
        use_container_width=True
    )

    st.markdown("**Conteo de SKUs por clase (acumulado de familias seleccionadas)**")
    cnt = (abc_sku_n2.groupby("Clase_ABC_SKU").size()
                     .reindex(["A","B","C"])
                     .rename("SKUs").fillna(0))
    df_cnt = cnt.reset_index().rename(columns={"Clase_ABC_SKU": "Clase"})
    chart_cnt = (
        alt.Chart(df_cnt)
        .mark_bar()
        .encode(
            x=alt.X("Clase:N", sort=["A","B","C"], title="Clase ABC"),
            y=alt.Y("SKUs:Q", scale=alt.Scale(domainMin=0), title="SKUs"),
            tooltip=["Clase:N", "SKUs:Q"]
        )
    )
    st.altair_chart(chart_cnt, use_container_width=True)

# --- ABC por SKU para TODAS las familias (no solo Nivel 2) ---
abc_sku_todos = abc_within_group(
    base, group_col="Familia_Key", key_col=col_sku,
    value_col="Ingresos_Row", a_cut=a_cut, b_cut=b_cut
).merge(fam_meta, on="Familia_Key", how="left").rename(columns={col_sku: "SKU"})

# =========================
# SÚPER ABC (ABC × XYZ) - opcional
# =========================
super_outputs = {}
if enable_super and month_cols:
    st.subheader("Súper ABC (ABC × XYZ)")

    # XYZ por Familias (unidades mensuales) usando Key
    fam_month = df[[fam_key_source] + month_cols].copy()
    fam_month["Familia_Key"] = normalize_text(fam_month[fam_key_source])
    xyz_fam = xyz_from_wide(fam_month, id_cols=["Familia_Key"], month_cols=month_cols,
                            cv_x=cv_x, cv_y=cv_y, level_name="Familia_Key")

    super_fam = (abc_fam[["Familia_Key","Clase_ABC","Ingresos_Total"]]
                 .merge(xyz_fam[["Familia_Key","CV","Clase_XYZ"]], on="Familia_Key", how="left")
                 .merge(fam_meta, on="Familia_Key", how="left"))
    super_fam["SuperABC"] = super_fam["Clase_ABC"].fillna("C") + super_fam["Clase_XYZ"].fillna("Z")

    fam_super_res = (super_fam.groupby("SuperABC", as_index=False)
                              .agg(Items=("SuperABC","count"),
                                   Ingresos=("Ingresos_Total","sum")))
    st.markdown("**Distribución Súper ABC (Familias)**")
    df_s1 = fam_super_res.copy()
    chart_s1 = (
        alt.Chart(df_s1)
        .mark_bar()
        .encode(
            x=alt.X("SuperABC:N", title="Categoría"),
            y=alt.Y("Items:Q", title="Items", scale=alt.Scale(domainMin=0)),
            tooltip=["SuperABC:N", "Items:Q", alt.Tooltip("Ingresos:Q", format=",.0f")]
        )
    )
    st.altair_chart(chart_s1, use_container_width=True)

    # XYZ por SKU dentro de familias seleccionadas (coherente con N2)
    sku_month = df[[fam_key_source, col_sku] + month_cols].copy()
    sku_month["Familia_Key"] = normalize_text(sku_month[fam_key_source])
    xyz_sku = xyz_from_wide(sku_month, id_cols=["Familia_Key", col_sku],
                            month_cols=month_cols, cv_x=cv_x, cv_y=cv_y, level_name="SKU")
    xyz_sku = xyz_sku.rename(columns={col_sku: "SKU"})

    if not sub_base.empty:
        sku_n2_keys = sub_base[["Familia_Key", col_sku]].drop_duplicates().rename(columns={col_sku: "SKU"})
        xyz_sku = xyz_sku.merge(sku_n2_keys, on=["Familia_Key","SKU"], how="inner")

    if not sub_base.empty and not abc_sku_n2.empty:
        tmp_abc = abc_sku_n2[["Familia_Key","SKU","Clase_ABC_SKU","Ingresos_Total"]]
        super_sku = (tmp_abc
                     .merge(xyz_sku[["Familia_Key","SKU","CV","Clase_XYZ"]],  # set to list to avoid error
                            on=["Familia_Key","SKU"], how="left")
                     .merge(fam_meta, on="Familia_Key", how="left"))
        # Fix for set usage:
        super_sku = (tmp_abc
                     .merge(xyz_sku[["Familia_Key","SKU","CV","Clase_XYZ"]],
                            on=["Familia_Key","SKU"], how="left")
                     .merge(fam_meta, on="Familia_Key", how="left"))

        super_sku["SuperABC"] = super_sku["Clase_ABC_SKU"].fillna("C") + super_sku["Clase_XYZ"].fillna("Z")

        sku_super_res = (super_sku.groupby("SuperABC", as_index=False)
                                   .agg(SKUs=("SuperABC","count"),
                                        Ingresos=("Ingresos_Total","sum")))
        st.markdown("**Distribución Súper ABC (SKUs en Familias Nivel 2)**")
        df_s2 = sku_super_res.copy()
        chart_s2 = (
            alt.Chart(df_s2)
            .mark_bar()
            .encode(
                x=alt.X("SuperABC:N", title="Categoría"),
                y=alt.Y("SKUs:Q", title="SKUs", scale=alt.Scale(domainMin=0)),
                tooltip=["SuperABC:N", "SKUs:Q", alt.Tooltip("Ingresos:Q", format=",.0f")]
            )
        )
        st.altair_chart(chart_s2, use_container_width=True)

        super_outputs.update({
            "XYZ_Familia": xyz_fam,
            "SuperABC_Familia": super_fam,
            "XYZ_SKU": xyz_sku,
            "SuperABC_SKU": super_sku,
            "Resumen_SuperABC_Familia": fam_super_res,
            "Resumen_SuperABC_SKU": sku_super_res
        })
    else:
        super_outputs.update({
            "XYZ_Familia": xyz_fam,
            "SuperABC_Familia": super_fam,
            "Resumen_SuperABC_Familia": fam_super_res
        })
elif enable_super and not month_cols:
    st.warning("Activa Súper ABC seleccionando al menos una columna de **mes** para calcular XYZ.")

# =========================
# Descarga Excel con resultados
# =========================
st.divider()
st.subheader("Descargar resultados (Excel)")

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    # Nivel 1
    abc_fam[cols_lvl1].to_excel(writer, index=False, sheet_name="ABC_Familia")
    fam_summary = res_fam.reset_index().rename(columns={"index":"Clase_ABC", "% Ingresos":"Pct_Ventas"})
    fam_summary.to_excel(writer, index=False, sheet_name="Resumen_Familia")

    # Nivel 2
    if not sub_base.empty and not abc_sku_n2.empty:
        abc_sku_n2.to_excel(writer, index=False, sheet_name="ABC_SKU_N2")
        if 'resumen_n2' in locals():
            resumen_n2.to_excel(writer, index=False, sheet_name="Conteos_SKU_por_Familia")

    # Todos los SKUs post-filtro
    abc_sku_todos.to_excel(writer, index=False, sheet_name="ABC_SKU_Todos")

    # Resumen de conteos clave
    resumen_conteos = pd.DataFrame([
        ["SKUs después de excluir estados", skus_total_filtrados],
        [f"SKUs en familias {','.join(clases_n2)} (Nivel 2)", skus_n2],
        ["Familias (tras filtros)", familias_total_filtradas],
    ], columns=["Métrica", "Valor"])
    resumen_conteos.to_excel(writer, index=False, sheet_name="Resumen_Conteos")

    # Súper ABC
    for name, df_out in super_outputs.items():
        try:
            df_out.to_excel(writer, index=False, sheet_name=name[:31])
        except Exception:
            alt_name = name.replace("_", "")[:31]
            df_out.to_excel(writer, index=False, sheet_name=alt_name)

    # Excluidos (si se usó el filtro)
    if col_desc != "<ninguna>":
        excl_info = pd.DataFrame({"Estado": [], "Filas_Excluidas": []})
        if not df_excluidos.empty:
            excl_info = (
                df_excluidos["_Estado_Articulo"]
                .value_counts()
                .rename_axis("Estado")
                .reset_index(name="Filas_Excluidas")
            )
            df_excluidos.to_excel(writer, index=False, sheet_name="Registros_Excluidos")
        excl_info.to_excel(writer, index=False, sheet_name="Excluidos_Info")

st.download_button(
    "⬇️ Descargar Excel (Resultados_ABC.xlsx)",
    data=buffer.getvalue(),
    file_name="Resultados_ABC.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.success("¡Listo! Gráficos con eje desde 0, conteos claros, ABC por Familias/SKU y Súper ABC (si se activó).")
