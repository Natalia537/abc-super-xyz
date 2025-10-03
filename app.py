# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import unicodedata, re
import altair as alt

# ================= Config =================
st.set_page_config(page_title="ABC / Súper ABC / Necesidades", layout="wide")
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

# --- Cobertura por Lead Time (interpolación) ---
_LT_POINTS = sorted([
    (0.0, 0.15),
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (1.566666667, 2.1),
    (2.0, 2.5),
    (2.5, 3.0),
    (3.0, 3.5),
    (3.5, 4.0),
], key=lambda x: x[0])

def coverage_from_lt(lt_months: float) -> float:
    try:
        x = float(lt_months)
    except Exception:
        return 1.0
    xs = [p[0] for p in _LT_POINTS]; ys = [p[1] for p in _LT_POINTS]
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0,x1 = xs[i-1], xs[i]; y0,y1 = ys[i-1], ys[i]
            t = (x - x0) / (x1 - x0) if x1!=x0 else 0
            return y0 + t*(y1-y0)
    return 1.0

def normalize_unit_type(v: str) -> str:
    s = str(v).strip().upper().replace("²","2").replace("M^2","M2")
    return "M2" if ("M2" in s or "M 2" in s) else "UNIDAD"

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

# ================ Carga ================
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

    st.markdown("---")
    st.markdown("**Necesidades de almacén (opcional, solo SKUs A)**")
    enable_wh = st.checkbox("Calcular **Necesidades de almacén** automáticas (UNIDAD / m2)")
    wh_type_col = wh_vol_col = lt_col = None
    wh_month_cols = []; lt_unit = "Días"; days_per_month = 30
    if enable_wh:
        wh_type_col = st.selectbox("Columna **Tipo de unidad** (valores: UNIDAD / m2)", cols)

        # Lead Time (columna + unidad)
        lt_candidates = [c for c in cols if "LT" in str(c).upper() or "LEAD" in str(c).upper()]
        lt_default_index = cols.index(lt_candidates[0]) if lt_candidates else 0
        lt_col = st.selectbox("Columna **Lead Time**", cols, index=lt_default_index)

        lt_unit = st.radio("Unidad de Lead Time", ["Meses", "Días"], index=1, horizontal=True)
        if lt_unit == "Días":
            days_per_month = st.number_input("Días por mes (para convertir LT a meses)", 1, 31, 30, 1)

        # Volumen (para m2)
        vol_candidates = [c for c in cols if "VOLUMEN" in str(c).upper()]
        v_idx = cols.index(vol_candidates[0]) if vol_candidates else 0
        wh_vol_col  = st.selectbox("Columna **Volumen** (m² por unidad, 999 = sin dimensión)",
                                   ["<ninguna>"]+cols, index=(v_idx+1 if v_idx or vol_candidates else 0))

        st.caption("Si no eliges meses, se **autodetectan** por nombre (ENE…DIC) y se usa el **promedio mensual**.")
        m_auto = detect_month_cols(cols)
        wh_month_cols = st.multiselect("Meses para demanda promedio (opcional)", cols, default=[c for c in m_auto][-12:])

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

# Meta familia
meta_cols=[]; 
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

# ABC SKU todos (para Excel)
abc_sku_todos = abc_within_group(base, "Familia_Key", col_sku, "Ingresos_Row", a_cut, b_cut)\
                    .merge(fam_meta, on="Familia_Key", how="left").rename(columns={col_sku:"SKU"})

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

# ================ Necesidades de almacén =================
volumen_df = pd.DataFrame()
wh_unid = pd.DataFrame(); wh_m2 = pd.DataFrame()
if enable_wh:
    if not wh_type_col:
        st.error("Selecciona la **columna de Tipo de unidad** (UNIDAD/m2).")
    else:
        # SKUs A (todas las familias)
        sku_a = abc_sku_todos[abc_sku_todos["Clase_ABC_SKU"]=="A"][["Familia_Key","SKU","Clase_ABC_SKU"]].copy()
        if sku_a.empty:
            st.warning("No hay SKUs A para calcular necesidades.")
        else:
            # Demanda mensual
            months = [c for c in (wh_month_cols or detect_month_cols(df.columns)) if c in df.columns]
            if months:
                tmp = df[[col_sku] + months].copy()
                for c in months: tmp[c] = to_num(tmp[c])
                dem = tmp.groupby(col_sku, as_index=False)[months].sum()
                dem["Demanda_Mensual"] = dem[months].mean(axis=1)
            else:
                st.info("No se detectaron columnas de meses → usando **unidades totales** como aproximación (1 mes).")
                dem = base.groupby(col_sku, as_index=False).agg(Demanda_Mensual=("Unid_Row","sum"))

            # Info: tipo + volumen
            info_cols = [col_sku, wh_type_col]
            if wh_vol_col and wh_vol_col!="<ninguna>": info_cols.append(wh_vol_col)
            info = df[info_cols].drop_duplicates()
            if wh_vol_col and wh_vol_col!="<ninguna>":
                info[wh_vol_col] = to_num(info[wh_vol_col])

            # Merge base
            wh = sku_a.merge(dem[[col_sku,"Demanda_Mensual"]], left_on="SKU", right_on=col_sku, how="left")\
                      .merge(info, left_on="SKU", right_on=col_sku, how="left")

            # --- Lead time → cobertura (convierte a meses si LT está en días) ---
            if lt_col:
                lt_series = df[[col_sku, lt_col]].drop_duplicates()
                lt_series[lt_col] = to_num(lt_series[lt_col])
                if lt_unit == "Días":
                    lt_series["_LT_meses"] = lt_series[lt_col] / float(days_per_month)
                else:
                    lt_series["_LT_meses"] = lt_series[lt_col]
                wh = wh.merge(lt_series[[col_sku, "_LT_meses"]], left_on="SKU", right_on=col_sku, how="left")
                wh["Cobertura_Meses"] = wh["_LT_meses"].apply(coverage_from_lt)
            else:
                wh["Cobertura_Meses"] = 1.0

            # Normaliza tipo, prepara volumen
            wh["Tipo_Unidad"] = wh[wh_type_col].apply(normalize_unit_type)
            if wh_vol_col and wh_vol_col in wh.columns:
                wh["Volumen_Unit"] = wh[wh_vol_col]
            else:
                wh["Volumen_Unit"] = np.nan

            # Necesidad por SKU
            wh["Demanda_Mensual"] = wh["Demanda_Mensual"].fillna(0)
            wh["Necesidad_Unidades"] = np.where(
                wh["Tipo_Unidad"].eq("UNIDAD"),
                wh["Demanda_Mensual"] * wh["Cobertura_Meses"], 0.0
            )

            wh["Is_Vol_999"] = np.where(
                (wh["Tipo_Unidad"].eq("M2")) & (wh["Volumen_Unit"]==999), 1, 0
            )
            wh["Necesidad_M2"] = np.where(
                (wh["Tipo_Unidad"].eq("M2")) & (~wh["Volumen_Unit"].isna()) & (wh["Volumen_Unit"]!=999),
                wh["Demanda_Mensual"] * wh["Cobertura_Meses"] * wh["Volumen_Unit"],
                0.0
            )

            # Separa vistas
            wh_unid = wh[wh["Tipo_Unidad"].eq("UNIDAD")].copy()
            wh_m2   = wh[wh["Tipo_Unidad"].eq("M2")].copy()

            # Totales + métricas
            total_unid = wh_unid["Necesidad_Unidades"].sum()
            total_m2   = wh_m2["Necesidad_M2"].sum()
            excl_999   = int(wh_m2["Is_Vol_999"].sum())
            c1,c2,c3 = st.columns(3)
            c1.metric("Necesidad total (UNIDADES)", f"{total_unid:,.0f}")
            c2.metric("Necesidad total (M2)", f"{total_m2:,.2f}")
            c3.metric("SKUs M2 con Volumen=999 (excluidos)", f"{excl_999:,}")
            st.caption(
                f"Cobertura calculada desde LT en **{lt_unit.lower()}**"
                f"{' (dividido entre ' + str(days_per_month) + ' días/mes)' if lt_unit=='Días' else ''} "
                "usando tu tabla (interpolación lineal)."
            )

            # Mostrar tablas (columnas seguras)
            cols_u = ["Familia_Key","SKU","Demanda_Mensual","Cobertura_Meses","Necesidad_Unidades"]
            cols_m = ["Familia_Key","SKU","Demanda_Mensual","Cobertura_Meses","Volumen_Unit","Necesidad_M2","Is_Vol_999"]
            st.subheader("Necesidades por SKU — **UNIDAD**")
            st.dataframe(
                wh_unid.reindex(columns=[c for c in cols_u if c in wh_unid.columns]).merge(fam_meta, on="Familia_Key", how="left"),
                use_container_width=True
            )
            st.subheader("Necesidades por SKU — **m2**")
            st.dataframe(
                wh_m2.reindex(columns=[c for c in cols_m if c in wh_m2.columns]).merge(fam_meta, on="Familia_Key", how="left"),
                use_container_width=True
            )

            # Volumen 999 listado
            volumen_df = wh_m2[wh_m2["Is_Vol_999"].eq(1)][["Familia_Key","SKU","Volumen_Unit"]].merge(fam_meta, on="Familia_Key", how="left")

# ================ Excel salida ================
st.divider(); st.subheader("Descargar resultados (Excel)")
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    abc_fam[cols_lvl1].to_excel(writer, index=False, sheet_name="ABC_Familia")
    (res_fam.reset_index().rename(columns={"index":"Clase_ABC","% Ingresos":"Pct_Ventas"})).to_excel(writer, index=False, sheet_name="Resumen_Familia")
    if not sub_base.empty and not abc_sku_n2.empty:
        abc_sku_n2.to_excel(writer, index=False, sheet_name="ABC_SKU_N2")
    abc_sku_todos.to_excel(writer, index=False, sheet_name="ABC_SKU_Todos")
    resumen_conteos = pd.DataFrame([
        ["SKUs post-filtro", skus_total_filtrados],
        [f"SKUs en familias {','.join(clases_n2)} (Nivel 2)", skus_n2],
        ["Familias post-filtro", familias_total_filtradas],
    ], columns=["Métrica","Valor"])
    resumen_conteos.to_excel(writer, index=False, sheet_name="Resumen_Conteos")
    for name, df_out in super_outputs.items():
        try: df_out.to_excel(writer, index=False, sheet_name=name[:31])
        except Exception: df_out.to_excel(writer, index=False, sheet_name=name.replace("_","")[:31])
    if not volumen_df.empty:
        volumen_df.to_excel(writer, index=False, sheet_name="Volumen_999")
    if not wh_unid.empty:
        wh_unid.to_excel(writer, index=False, sheet_name="Necesidades_UNIDAD")
    if not wh_m2.empty:
        wh_m2.to_excel(writer, index=False, sheet_name="Necesidades_M2")
    if not df_excluidos.empty:
        excl_info = df_excluidos["_Estado_Articulo"].value_counts().rename_axis("Estado").reset_index(name="Filas_Excluidas")
        excl_info.to_excel(writer, index=False, sheet_name="Excluidos_Info")
        df_excluidos.to_excel(writer, index=False, sheet_name="Registros_Excluidos")

st.download_button("⬇️ Descargar Excel (Resultados_ABC.xlsx)",
                   data=buffer.getvalue(),
                   file_name="Resultados_ABC.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("¡Listo! ABC/Súper ABC y Necesidades automáticas por UNIDAD/m2 con cobertura por LT (días o meses).")
