# app.py — ABC / Súper ABC (ABC×XYZ) con STOCK en tablas, gráficos y Excel
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import unicodedata, re
import altair as alt

# ================= Config =================
st.set_page_config(page_title="ABC / Súper ABC", layout="wide")
st.title("ABC / Súper ABC (ABC×XYZ) por Familias y SKUs con STOCK")

with st.expander("ℹ️ Guía rápida"):
    st.markdown("""
**ABC (Pareto por ventas)** → A/B/C según % acumulado de ventas.  
**XYZ (variabilidad)** → CV (σ/μ) con columnas de **meses (unidades)** → X/Y/Z.  
**Súper ABC** = combinación (p.ej. AX = A en ventas + X estable).  
**Stock (naranja)** → muestra existencias totales de Familias o SKUs en tablas, gráficos y Excel.  
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
           .agg(Unid_Total=("Unid_Row","sum"),
                Ingresos_Total=(value_col,"sum"),
                Stock_Total=("Stock_Row","sum")))
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
    col_stock    = st.selectbox("Columna **Stock** (opcional)", ["<ninguna>"]+cols, index=0)

    st.markdown("**Filtro por estado en descripción (opcional)**")
    col_desc = st.selectbox("Columna con (D)/(DXF)/(P.P)", ["<ninguna>"]+cols, index=0)
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
        candidates = [c for c in cols if c not in {col_sku,col_fam_name,col_fam_code,col_unid,col_price,col_stock} and c!="<ninguna>"]
        st.info("Selecciona columnas de **meses (unidades)** para XYZ.")
        month_cols = st.multiselect("Columnas de meses (UNIDADES)", candidates)
        cv_x = st.number_input("Umbral X (CV ≤ X)", 0.0, 1.0, 0.25, 0.01)
        cv_y = st.number_input("Umbral Y (CV ≤ Y)", cv_x, 1.0, 0.50, 0.01)

    st.markdown("**Gráficos — Escala de Stock**")
    scale_stock_thousands = st.checkbox("Mostrar Stock ÷ 1000 en gráficos", value=True)

    submitted = st.form_submit_button("Calcular")
if not submitted: st.stop()

# ================ Filtros y limpieza ================
if col_desc!="<ninguna>" and exclude_status:
    df["_Estado_Articulo"] = df[col_desc].apply(extract_item_status)
    mask = df["_Estado_Articulo"].isin(exclude_status)
    df = df[~mask].copy()
else:
    df["_Estado_Articulo"]=None

use_code = (col_fam_code!="<ninguna>")
fam_key_source = col_fam_code if use_code else (col_fam_name if col_fam_name!="<ninguna>" else None)
if fam_key_source is None: st.error("Debes elegir una columna de Familia."); st.stop()

# ================ Base agrupada ================
base = df[[col_sku, fam_key_source, col_unid, col_price] + ([col_stock] if col_stock!="<ninguna>" else [])].copy()
base["Familia_Key"] = normalize_text(base[fam_key_source])
base["Unid_Row"] = to_num(base[col_unid])
base["Precio_Row"] = to_num(base[col_price])
base["Ingresos_Row"] = base["Unid_Row"] * base["Precio_Row"]
base["Stock_Row"] = to_num(base[col_stock]) if col_stock!="<ninguna>" else 0

# ================ Nivel 1: ABC Familias ================
fam_agg = (base.groupby("Familia_Key", as_index=False)
           .agg(Unid_Total=("Unid_Row","sum"),
                Ingresos_Total=("Ingresos_Row","sum"),
                Stock_Total=("Stock_Row","sum")))
abc_fam = abc_from_values(fam_agg, "Ingresos_Total", a_cut, b_cut)
cols_lvl1 = ["Familia_Key","Unid_Total","Ingresos_Total","Stock_Total","%Ingresos","%Acum","Clase_ABC"]
st.subheader("Nivel 1: ABC Familias")
st.dataframe(abc_fam[cols_lvl1], use_container_width=True)

# === Resumen gráfico ABC Familias ===
res_fam = abc_fam.groupby("Clase_ABC", as_index=False).agg(Ingresos=("Ingresos_Total","sum"),Stock=("Stock_Total","sum"))
res_fam["Pct_Ventas"] = (res_fam["Ingresos"]/res_fam["Ingresos"].sum())*100
plot_fam = res_fam.copy();  plot_fam["Clase"] = plot_fam["Clase_ABC"]
if scale_stock_thousands: plot_fam["Stock"] /= 1000
plot_fam_long = plot_fam.melt(id_vars=["Clase"], value_vars=["Pct_Ventas","Stock"], var_name="Métrica", value_name="Valor")
st.altair_chart(
    alt.Chart(plot_fam_long).mark_bar().encode(
        x=alt.X("Clase:N", sort=["A","B","C"]),
        xOffset="Métrica:N",
        y="Valor:Q",
        color=alt.Color("Métrica:N", scale=alt.Scale(domain=["Pct_Ventas","Stock"], range=["#4C78A8","#FFA500"])),
        tooltip=["Clase:N","Métrica:N","Valor:Q"]
    ).properties(title="ABC Familias: % Ventas vs Stock"),
    use_container_width=True
)

# ================ Nivel 2: ABC SKU ================
fams_sel = abc_fam.loc[abc_fam["Clase_ABC"].isin(clases_n2),"Familia_Key"]
sub_base = base[base["Familia_Key"].isin(fams_sel)]
abc_sku_n2 = abc_within_group(sub_base, "Familia_Key", col_sku, "Ingresos_Row", a_cut, b_cut)
st.subheader("Nivel 2: ABC SKU (familias seleccionadas)")
st.dataframe(abc_sku_n2,use_container_width=True)

sku_res = abc_sku_n2.groupby("Clase_ABC_SKU",as_index=False).agg(SKUs=("Familia_Key","count"),Stock=("Stock_Total","sum"))
sku_res["Clase"]=sku_res["Clase_ABC_SKU"]
if scale_stock_thousands: sku_res["Stock"]/=1000
plot_sku_long = sku_res.melt(id_vars=["Clase"],value_vars=["SKUs","Stock"],var_name="Métrica",value_name="Valor")
st.altair_chart(
    alt.Chart(plot_sku_long).mark_bar().encode(
        x=alt.X("Clase:N",sort=["A","B","C"]),xOffset="Métrica:N",y="Valor:Q",
        color=alt.Color("Métrica:N",scale=alt.Scale(domain=["SKUs","Stock"],range=["#4C78A8","#FFA500"])),
        tooltip=["Clase:N","Métrica:N","Valor:Q"]
    ).properties(title="ABC SKU (N2): SKUs vs Stock"),
    use_container_width=True
)

# ================ Súper ABC ================
if enable_super and month_cols:
    st.subheader("Súper ABC (ABC × XYZ)")
    fam_month=df[[fam_key_source]+month_cols].copy(); fam_month["Familia_Key"]=normalize_text(fam_month[fam_key_source])
    xyz_fam=xyz_from_wide(fam_month,["Familia_Key"],month_cols,cv_x,cv_y,"Familia_Key")
    stock_fam=base.groupby("Familia_Key",as_index=False)["Stock_Row"].sum().rename(columns={"Stock_Row":"Stock_Total"})
    xyz_fam=xyz_fam.merge(stock_fam,on="Familia_Key",how="left")
    super_fam=abc_fam.merge(xyz_fam,on="Familia_Key",how="left")
    super_fam["SuperABC"]=super_fam["Clase_ABC"]+super_fam["Clase_XYZ"]
    fam_super_res=super_fam.groupby("SuperABC",as_index=False).agg(Items=("SuperABC","count"),Stock=("Stock_Total","sum"))
    if scale_stock_thousands: fam_super_res["Stock"]/=1000
    fam_super_long=fam_super_res.melt(id_vars=["SuperABC"],value_vars=["Items","Stock"],var_name="Métrica",value_name="Valor")
    st.altair_chart(
        alt.Chart(fam_super_long).mark_bar().encode(
            x="SuperABC:N",xOffset="Métrica:N",y="Valor:Q",
            color=alt.Color("Métrica:N",scale=alt.Scale(domain=["Items","Stock"],range=["#4C78A8","#FFA500"])),
            tooltip=["SuperABC:N","Métrica:N","Valor:Q"]
        ).properties(title="Súper ABC (Familias): Items vs Stock"),
        use_container_width=True
    )

# ================ Excel salida ================
st.divider(); st.subheader("Descargar resultados (Excel)")
buffer=BytesIO()
with pd
