# app.py  (sin módulo de Necesidades de almacén)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import unicodedata, re
import altair as alt

# ================= Config =================
st.set_page_config(page_title="ABC / Súper ABC", layout="wide")
st.title("ABC / Súper ABC (ABC×XYZ) por Familias y SKUs")

with st.expander("ℹ️ Guía rápida"):
    st.markdown("""
**ABC (Pareto por ventas)** → A/B/C según % acumulado de ventas.  
**XYZ (variabilidad)** → CV (σ/μ) con columnas de **meses (unidades)** → X/Y/Z.  
**Súper ABC** = combinación (p.ej. AX = A en ventas + X estable).  
Los gráficos empiezan en **0** y tienen *tooltips*.
""")

# ================ Utilidades ================
def safe_block(label, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        with st.expander(f"⚠️ Algo salió mal en: {label}. Ver detalles técnicos"):
            st.exception(e)
        st.error(f"Algo salió mal en **{label}**. Revisa datos o parámetros.")
        return None

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str)
                         .str.replace(r"[^\d\-,\.]", "", regex=True)
                         .str.replace(",", "", regex=False),
                         errors="coerce").fillna(0)

def normalize_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("ascii"))
    return s.str.upper()

def pick_mode(s: pd.Series) -> str:
    s = s.astype(str).str.strip()
    s = s[(s!="") & s.notna()]
    return "" if s.empty else s.value_counts().idxmax()

def abc_from_values(df, value_col, a_cut=80.0, b_cut=95.0):
    out = df.copy().sort_values([value_col], ascending=[False], kind="mergesort").reset_index(drop=True)
    total = out[value_col].sum()
    if total <= 0:
        out["%Ingresos"]=0.0; out["%Acum"]=0.0; out["Clase_ABC"]="C"; return out
    out["%Ingresos"] = (out[value_col]/total)*100
    out["%Acum"] = out["%Ingresos"].cumsum()
    out["Clase_ABC"] = out["%Acum"].apply(lambda p: "A" if p<=a_cut else ("B" if p<=b_cut else "C"))
    return out

def abc_within_group(df, group_col, key_col, value_col, a_cut=80.0, b_cut=95.0):
    g = (df.groupby([group_col, key_col], as_index=False)
           .agg(Unid_Total=("Unid_Row","sum"), Ingresos_Total=(value_col,"sum")))
    parts=[]
    for fam, sub in g.groupby(group_col, sort=False):
        sub = sub.sort_values(["Ingresos_Total", key_col], ascending=[False, True], kind="mergesort").copy()
        t = sub["Ingresos_Total"].sum()
        if t<=0:
            sub["%Ingresos_Fam"]=0.0; sub["%Acum_Fam"]=0.0; sub["Clase_ABC_SKU"]="C"
        else:
            sub["%Ingresos_Fam"] = (sub["Ingresos_Total"]/t)*100
            sub["%Acum_Fam"] = sub["%Ingresos_Fam"].cumsum()
            sub["Clase_ABC_SKU"] = sub["%Acum_Fam"].apply(lambda p: "A" if p<=a_cut else ("B" if p<=b_cut else "C"))
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)

def xyz_from_wide(df, id_cols, month_cols, cv_x=0.25, cv_y=0.50, level_name="Entidad"):
    mtx = df[id_cols + month_cols].copy()
    for c in month_cols: mtx[c] = to_num(mtx[c])
    agg = mtx.groupby(id_cols, as_index=False)[month_cols].sum()
    vals = agg[month_cols].to_numpy(dtype=float)
    mean = vals.mean(axis=1); std = vals.std(axis=1, ddof=0)
    cv = np.where(mean>0, std/mean, np.inf)
    out = agg[id_cols].copy(); out["CV"]=cv
    out["Clase_XYZ"]=out["CV"].apply(lambda c: "X" if c<=cv_x else ("Y" if c<=cv_y else "Z"))
    if level_name: out = out.rename(columns={id_cols[-1]:level_name})
    return out

def extract_item_status(text:str):
    s = str(text).strip()
    m = re.search(r"\(\s*(P\.?P|DXF|D)\s*\)\s*$", s, flags=re.IGNORECASE)
    if not m: return None
    token = m.group(1).upper().replace(".","")
    return {"PP":"P.P", "DXF":"DXF", "D":"D"}.get(token, None)

# --- Detección de columnas de meses (ES/EN) ---
_MONTH_MAP = {
    "ENE":1,"ENERO":1,"JAN":1,"JANUARY":1,
    "FEB":2,"FEBRERO":2,"FEBRUARY":2,
    "MAR":3,"MARZO":3,"MARCH":3,
    "ABR":4,"ABRIL":4,"APR":4,"APRIL":4,
    "MAY":5,"MAYO":5,"MAY":5,
    "JUN":6,"JUNIO":6,"JUNE":6,
    "JUL":7,"JULIO":7,"JULY":7,
    "AGO":8,"AGOSTO":8,"AUG":8,"AUGUST":8,
    "SEP":9,"SEPT":9,"SET":9,"SEPTIEMBRE":9,"SEPTEMBER":9,
    "OCT":10,"OCTUBRE":10,"OCTOBER":10,
    "NOV":11,"NOVIEMBRE":11,"NOVEMBER":11,
    "DIC":12,"DICIEMBRE":12,"DEC":12,"DECEMBER":12,
}
def detect_month_cols(df_cols):
    found=[]
    for c in df_cols:
        up = str(c).strip().upper()
        for key, m in _MONTH_MAP.items():
            if key in up: found.append((c,m)); break
    found = sorted(found, key=lambda x: (x[1], list(df_cols).index(x[0])))
    return [c for c,_ in found]

# ================ Carga ================
@st.cache_data(show_spinner=False)
def listar_hojas(file_bytes:bytes, ext:str):
    if ext=="xlsx":
        return pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl").sheet_names
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names

@st.cache_data(show_spinner=False)
def leer_df(file_bytes:bytes, ext:str, sheet:str|None):
    if ext=="xlsx":
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet, engine="openpyxl")
    elif ext=="xls":
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
    try:
        return pd.read_csv(BytesIO(file_bytes), engine="pyarrow")
    except Exception:
        return pd.read_csv(BytesIO(file_bytes))

archivo = st.file_uploader("Sube un Excel (.xlsx/.xls) o CSV", type=["xlsx","xls","csv"])
if not archivo: st.info("👉 Sube tu archivo para comenzar."); st.stop()
ext = archivo.name.split(".")[-1].lower(); file_bytes = archivo.getvalue()
if ext in ("xlsx","xls"):
    hoja = st.selectbox("Elige la hoja", safe_block("Listar hojas", listar_hojas, file_bytes, ext))
    df = safe_block("Leer datos", leer_df, file_bytes, ext, hoja)
else:
    df = safe_block("Leer datos", leer_df, file_bytes, ext, None)
if df is None: st.stop()
st.subheader("Vista previa de datos"); st.dataframe(df.head(), use_container_width=True)

# ================ Parámetros ================
with st.form("params"):
    cols = list(df.columns)
    col_sku      = st.selectbox("Columna **Código / SKU**", cols)
    col_fam_name = st.selectbox("Columna **Nombre de Familia** (recomendado)", ["<ninguna>"]+cols, index=0)
    col_fam_code = st.selectbox("Columna **Código de Familia** (opcional)", ["<ninguna>"]+cols, index=0)
    col_unid     = st.selectbox("Columna **Unidades vendidas (detalle)**", cols)
    col_price    = st.selectbox("Columna **Precio/Monto unitario**", cols)

    st.markdown("**Filtro por estado en descripción (opcional)**")
    col_desc = st.selectbox("Columna con (D)/(DXF)/(P.P) al final (ej. D.Articulo)", ["<ninguna>"]+cols, index=0)
    exclude_status = st.multiselect("Excluir estos estados", options=["D","DXF","P.P"])

    st.markdown("**Parámetros ABC**")
    a_cut = st.slider("Corte A (%)", 50, 95, 80)
    b_cut = st.slider("Corte B (%)", a_cut+1, 99, 95)

    st.markdown("**Nivel 2 — Familias a profundizar**")
    clases_n2 = st.multiselect("Clases de familias", options=["A","B","C"], default=["A"])

    st.markdown("**Súper ABC (opcional)**")
    enable_super = st.checkbox("Activar Súper ABC (combinar ABC con XYZ)")
    month_cols = []; cv_x=0.25; cv_y=0.50
    if enable_super:
        candidates = [c for c in cols if c not in {col_sku,col_fam_name,col_fam_code,col_unid,col_price} and c!="<ninguna>"]
        st.info("Selecciona columnas de **meses** (unidades) para calcular XYZ.")
        month_cols = st.multiselect("Columnas de meses (UNIDADES)", candidates)
        cv_x = st.number_input("Umbral X (CV ≤ X)", 0.0, 1.0, 0.25, 0.01)
        cv_y = st.number_input("Umbral Y (CV ≤ Y)", cv_x, 1.0, 0.50, 0.01)

    submitted = st.form_submit_button("Calcular")
if not submitted: st.stop()

# ================ Filtro estados ================
df_excluidos = pd.DataFrame()
if col_desc!="<ninguna>" and exclude_status:
    df["_Estado_Articulo"] = df[col_desc].apply(extract_item_status)
    mask = df["_Estado_Articulo"].isin(exclude_status)
    df_excluidos = df[mask].copy()
    df = df[~mask].copy()
    if not df_excluidos.empty:
        st.info("Excluidos: " + ", ".join(f"{k}: {v}" for k,v in df_excluidos["_Estado_Articulo"].value_counts().to_dict().items()))
else:
    df["_Estado_Articulo"]=None

# ================ Base agrupada ================
use_code = (col_fam_code!="<ninguna>")
fam_key_source = col_fam_code if use_code else (col_fam_name if col_fam_name!="<ninguna>" else None)
if fam_key_source is None:
    st.error("Debes seleccionar al menos una columna de **Familia** (Nombre o Código)."); st.stop()

base = df[[col_sku, fam_key_source, col_unid, col_price]].copy()
base[col_sku] = base[col_sku].astype(str).str.strip()
base[fam_key_source] = base[fam_key_source].astype(str).str.strip()
base = base[base[col_sku].ne("") & base[fam_key_source].ne("")]
base["Familia_Key"] = normalize_text(base[fam_key_source])
base["Unid_Row"]   = to_num(base[col_unid])
base["Precio_Row"] = to_num(base[col_price])
base["Ingresos_Row"] = base["Unid_Row"] * base["Precio_Row"]

skus_total_filtrados = base[col_sku].nunique()
familias_total_filtradas = base["Familia_Key"].nunique()
st.info(f"SKUs post-filtro: **{skus_total_filtrados:,}** | Familias: **{familias_total_filtradas:,}**")

# Meta familia (código/nombre)
meta_cols=[]
if col_fam_code!="<ninguna>": meta_cols.append(col_fam_code)
if col_fam_name!="<ninguna>": meta_cols.append(col_fam_name)
if fam_key_source not in meta_cols: meta_cols.append(fam_key_source)
fam_meta_raw = df[meta_cols].copy(); fam_meta_raw["Familia_Key"] = normalize_text(fam_meta_raw[fam_key_source])
if col_fam_name!="<ninguna>": name_source=col_fam_name
elif col_fam_code=="<ninguna>": name_source=fam_key_source
else: name_source=None
if col_fam_code!="<ninguna>" and name_source:
    fam_meta = (fam_meta_raw.groupby("Familia_Key", as_index=False)
                .agg(Familia_Cod=(col_fam_code, pick_mode), Familia_Nombre=(name_source, pick_mode)))
elif col_fam_code!="<ninguna>" and not name_source:
    fam_meta = (fam_meta_raw.groupby("Familia_Key", as_index=False)
                .agg(Familia_Cod=(col_fam_code, pick_mode))); fam_meta["Familia_Nombre"]=""
else:
    fam_meta = (fam_meta_raw.groupby("Familia_Key", as_index=False)
                .agg(Familia_Nombre=(name_source, pick_mode))); fam_meta["Familia_Cod"]=""

# ================ NIVEL 1: ABC Familias ================
fam_agg = (base.groupby("Familia_Key", as_index=False)
                .agg(Unid_Total=("Unid_Row","sum"), Ingresos_Total=("Ingresos_Row","sum")))
abc_fam = abc_from_values(fam_agg, "Ingresos_Total", a_cut, b_cut).merge(fam_meta, on="Familia_Key", how="left")
cols_lvl1 = ["Familia_Key","Familia_Cod","Familia_Nombre","Unid_Total","Ingresos_Total","%Ingresos","%Acum","Clase_ABC"]
st.subheader("Nivel 1: ABC por **Familias** (ventas)")
st.dataframe(abc_fam[cols_lvl1], use_container_width=True)

res_fam = (abc_fam.groupby("Clase_ABC", as_index=False).agg(Ingresos=("Ingresos_Total","sum")))
res_fam["% Ingresos"] = (res_fam["Ingresos"]/res_fam["Ingresos"].sum())*100
res_fam = res_fam.set_index("Clase_ABC").reindex(["A","B","C"]).fillna(0)
df_fam_plot = res_fam.reset_index().rename(columns={"% Ingresos":"Pct","Clase_ABC":"Clase"})
st.altair_chart(
    alt.Chart(df_fam_plot).mark_bar().encode(
        x=alt.X("Clase:N", sort=["A","B","C"], title="Clase ABC"),
        y=alt.Y("Pct:Q", scale=alt.Scale(domainMin=0, domainMax=100), title="% de ventas"),
        tooltip=["Clase:N", alt.Tooltip("Pct:Q", format=".1f")]
    ),
    use_container_width=True
)

# ================ NIVEL 2: ABC SKU ================
fams_sel_keys = abc_fam.loc[abc_fam["Clase_ABC"].isin(clases_n2), "Familia_Key"].unique().tolist()
sub_base = base[base["Familia_Key"].isin(fams_sel_keys)].copy()
skus_n2 = sub_base[col_sku].nunique() if not sub_base.empty else 0
st.info(f"SKUs en familias {clases_n2}: **{skus_n2:,}**")

if sub_base.empty:
    st.warning("No hay filas para Nivel 2.")
    abc_sku_n2 = pd.DataFrame()
else:
    abc_sku_n2 = abc_within_group(sub_base, "Familia_Key", col_sku, "Ingresos_Row", a_cut, b_cut)
    abc_sku_n2 = abc_sku_n2.merge(fam_meta, on="Familia_Key", how="left").rename(columns={col_sku:"SKU"})
    st.subheader(f"Nivel 2: ABC por **SKU** (familias {clases_n2})")
    st.dataframe(abc_sku_n2[["Familia_Key","Familia_Cod","Familia_Nombre","SKU","Unid_Total","Ingresos_Total","%Ingresos_Fam","%Acum_Fam","Clase_ABC_SKU"]],
                 use_container_width=True)
    cnt = (abc_sku_n2.groupby("Clase_ABC_SKU").size().reindex(["A","B","C"]).rename("SKUs").fillna(0))
    df_cnt = cnt.reset_index().rename(columns={"Clase_ABC_SKU":"Clase"})
    st.altair_chart(
        alt.Chart(df_cnt).mark_bar().encode(
            x=alt.X("Clase:N", sort=["A","B","C"], title="Clase ABC"),
            y=alt.Y("SKUs:Q", scale=alt.Scale(domainMin=0), title="SKUs"),
            tooltip=["Clase:N", "SKUs:Q"]
        ), use_container_width=True
    )

# ABC por SKU (todos) – útil para filtros/export
abc_sku_todos = abc_within_group(base, "Familia_Key", col_sku, "Ingresos_Row", a_cut, b_cut)\
                    .merge(fam_meta, on="Familia_Key", how="left").rename(columns={col_sku:"SKU"})

# =========================
# Ventas vs Demanda (mensual) — usa meses de Súper ABC (o autodetección)
# =========================
st.subheader("Ventas vs Demanda (mensual)")

if enable_super and month_cols:
    months_for_chart = [c for c in month_cols if c in df.columns]
    st.caption("Usando las mismas columnas de meses seleccionadas en **Súper ABC**.")
else:
    auto_months = detect_month_cols(df.columns)
    months_for_chart = [c for c in auto_months if c in df.columns][-12:]
    st.caption("No seleccionaste meses en Súper ABC; se **autodetectan** por nombre (ENE…DIC).")

col1, col2 = st.columns(2)
only_skus_A  = col1.checkbox("Solo SKUs clase A", value=False, key="vvd_onlyA")
only_fams_n2 = col2.checkbox("Solo Familias de Nivel 2", value=False, key="vvd_onlyN2")

ventas_vs_demanda_df = None
if months_for_chart:
    df_plot = df.copy()

    if only_fams_n2:
        if fams_sel_keys:
            df_plot["_FK"] = normalize_text(df_plot[fam_key_source].astype(str))
            df_plot = df_plot[df_plot["_FK"].isin(fams_sel_keys)]

    if only_skus_A and not abc_sku_todos.empty:
        skus_A = set(abc_sku_todos.loc[abc_sku_todos["Clase_ABC_SKU"]=="A", "SKU"].astype(str))
        df_plot = df_plot[df_plot[col_sku].astype(str).isin(skus_A)]

    if col_price in df_plot.columns:
        precio_unit = to_num(df_plot[col_price])
        filas=[]
        for m in months_for_chart:
            if m not in df_plot.columns: continue
            unid_m = to_num(df_plot[m]).sum()
            ventas_m = (to_num(df_plot[m]) * precio_unit).sum()
            up = str(m).strip().upper()
            mes_num = next((v for k,v in _MONTH_MAP.items() if k in up), None)
            filas.append({"Mes":str(m),"MesNum":mes_num if mes_num is not None else 999,
                          "Demanda":float(unid_m),"Ventas":float(ventas_m)})
        if filas:
            season_df = pd.DataFrame(filas).sort_values(["MesNum","Mes"]).reset_index(drop=True)
            base_ch = alt.Chart(season_df).encode(x=alt.X("Mes:N", title="Mes"))
            bars = base_ch.mark_bar().encode(
                y=alt.Y("Ventas:Q", axis=alt.Axis(title="Ventas (valor)"),
                        scale=alt.Scale(domainMin=0)),
                tooltip=[alt.Tooltip("Mes:N"),
                         alt.Tooltip("Ventas:Q", format=",.0f"),
                         alt.Tooltip("Demanda:Q", format=",.0f")]
            )
            line = base_ch.mark_line(point=True).encode(
                y=alt.Y("Demanda:Q", axis=alt.Axis(title="Unidades"),
                        scale=alt.Scale(domainMin=0)),
                tooltip=[alt.Tooltip("Mes:N"),
                         alt.Tooltip("Demanda:Q", format=",.0f"),
                         alt.Tooltip("Ventas:Q", format=",.0f")]
            )
            st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent'),
                            use_container_width=True)
            with st.expander("Ver tabla (Ventas vs Demanda)"):
                st.dataframe(season_df[["Mes","Ventas","Demanda"]], use_container_width=True)
            ventas_vs_demanda_df = season_df[["Mes","Ventas","Demanda"]].copy()
else:
    st.warning("No hay columnas de meses disponibles para construir la serie.")

# ================ Súper ABC (opcional) ================
super_outputs={}
if enable_super and month_cols:
    st.subheader("Súper ABC (ABC × XYZ)")
    fam_month = df[[fam_key_source]+month_cols].copy(); fam_month["Familia_Key"]=normalize_text(fam_month[fam_key_source])
    xyz_fam = xyz_from_wide(fam_month, ["Familia_Key"], month_cols, cv_x, cv_y, "Familia_Key")
    super_fam = (abc_fam[["Familia_Key","Clase_ABC","Ingresos_Total"]]
                 .merge(xyz_fam[["Familia_Key","CV","Clase_XYZ"]], on="Familia_Key", how="left")
                 .merge(fam_meta, on="Familia_Key", how="left"))
    super_fam["SuperABC"] = super_fam["Clase_ABC"].fillna("C")+super_fam["Clase_XYZ"].fillna("Z")
    fam_super_res = super_fam.groupby("SuperABC", as_index=False).agg(Items=("SuperABC","count"),
                                                                      Ingresos=("Ingresos_Total","sum"))
    st.altair_chart(
        alt.Chart(fam_super_res).mark_bar().encode(
            x=alt.X("SuperABC:N", title="Categoría"),
            y=alt.Y("Items:Q", scale=alt.Scale(domainMin=0), title="Items"),
            tooltip=["SuperABC:N","Items:Q", alt.Tooltip("Ingresos:Q", format=",.0f")]
        ), use_container_width=True
    )

    sku_month = df[[fam_key_source, col_sku]+month_cols].copy(); sku_month["Familia_Key"]=normalize_text(sku_month[fam_key_source])
    xyz_sku = xyz_from_wide(sku_month, ["Familia_Key", col_sku], month_cols, cv_x, cv_y, "SKU").rename(columns={col_sku:"SKU"})
    if not sub_base.empty:
        sku_n2_keys = sub_base[["Familia_Key", col_sku]].drop_duplicates().rename(columns={col_sku:"SKU"})
        xyz_sku = xyz_sku.merge(sku_n2_keys, on=["Familia_Key","SKU"], how="inner")
    if not sub_base.empty and not abc_sku_n2.empty:
        tmp_abc = abc_sku_n2[["Familia_Key","SKU","Clase_ABC_SKU","Ingresos_Total"]]
        super_sku = (tmp_abc.merge(xyz_sku[["Familia_Key","SKU","CV","Clase_XYZ"]],
                                   on=["Familia_Key","SKU"], how="left")
                             .merge(fam_meta, on="Familia_Key", how="left"))
        super_sku["SuperABC"]=super_sku["Clase_ABC_SKU"].fillna("C")+super_sku["Clase_XYZ"].fillna("Z")
        sku_super_res = super_sku.groupby("SuperABC", as_index=False).agg(SKUs=("SuperABC","count"),
                                                                          Ingresos=("Ingresos_Total","sum"))
        st.altair_chart(
            alt.Chart(sku_super_res).mark_bar().encode(
                x=alt.X("SuperABC:N", title="Categoría"),
                y=alt.Y("SKUs:Q", scale=alt.Scale(domainMin=0), title="SKUs"),
                tooltip=["SuperABC:N","SKUs:Q", alt.Tooltip("Ingresos:Q", format=",.0f")]
            ), use_container_width=True
        )
        super_outputs.update({
            "XYZ_Familia": xyz_fam, "SuperABC_Familia": super_fam,
            "XYZ_SKU": xyz_sku, "SuperABC_SKU": super_sku,
            "Resumen_SuperABC_Familia": fam_super_res, "Resumen_SuperABC_SKU": sku_super_res
        })
    else:
        super_outputs.update({"XYZ_Familia": xyz_fam, "SuperABC_Familia": super_fam,
                              "Resumen_SuperABC_Familia": fam_super_res})

# ================ Excel salida ================
st.divider()
st.subheader("Descargar resultados (Excel)")

buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    # Nivel 1
    abc_fam[cols_lvl1].to_excel(writer, index=False, sheet_name="ABC_Familia")
    (res_fam.reset_index()
           .rename(columns={"index": "Clase_ABC", "% Ingresos": "Pct_Ventas"})
     ).to_excel(writer, index=False, sheet_name="Resumen_Familia")

    # Nivel 2
    if not sub_base.empty and not abc_sku_n2.empty:
        abc_sku_n2.to_excel(writer, index=False, sheet_name="ABC_SKU_N2")
    abc_sku_todos.to_excel(writer, index=False, sheet_name="ABC_SKU_Todos")

    # Resumen de conteos
    resumen_conteos = pd.DataFrame(
        [
            ["SKUs post-filtro", skus_total_filtrados],
            [f"SKUs en familias {','.join(clases_n2)} (Nivel 2)", skus_n2],
            ["Familias post-filtro", familias_total_filtradas],
        ],
        columns=["Métrica", "Valor"],
    )
    resumen_conteos.to_excel(writer, index=False, sheet_name="Resumen_Conteos")

    # Súper ABC (si existe)
    for name, df_out in super_outputs.items():
        try:
            df_out.to_excel(writer, index=False, sheet_name=name[:31])
        except Exception:
            df_out.to_excel(writer, index=False, sheet_name=name.replace("_", "")[:31])

    # Ventas vs Demanda
    if isinstance(ventas_vs_demanda_df, pd.DataFrame):
        ventas_vs_demanda_df.to_excel(writer, index=False, sheet_name="Ventas_vs_Demanda")

buffer.seek(0)

st.download_button(
    "⬇️ Descargar Excel (Resultados_ABC.xlsx)",
    data=buffer.getvalue(),
    file_name="Resultados_ABC.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("¡Listo! ABC/Súper ABC y gráfico Ventas vs Demanda (sin módulo de almacén).")
