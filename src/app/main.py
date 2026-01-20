import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# --- 1. ÎNCĂRCARE RESURSE ---
@st.cache_resource
def load_resources():
    model_path = r"D:\Facultate\RN\models\trained_model.keras"
    config_path = r"D:\Facultate\RN\config\scaler.skl"
    
    if os.path.exists(model_path) and os.path.exists(config_path):
        model = tf.keras.models.load_model(model_path)
        scaler_cfg = joblib.load(config_path)
        return model, scaler_cfg
    return None, None

# --- 2. CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="SIA Diagnostic Expert", page_icon="🔬", layout="wide")

model, scaler_cfg = load_resources()

if not model:
    st.error("⚠️ Modelul nu a fost găsit. Rulează mai întâi scriptul de antrenare.")
    st.stop()

# --- 3. LISTA DE ÎNTREBĂRI ---
questions = [
    {"id": "Q1", "text": "Cât de ridicată este febra?", "options": ["Normală (T)", "Ușoară (P)", "Moderată (P)", "Ridicată (P)", "Foarte ridicată (P)"]},
    {"id": "Q2", "text": "Cât de capabil te simți să întreprinzi activități normale?", "options": ["Foarte puțin (P)", "Puțin (P)", "Moderat (P)", "Mult (T)", "Foarte mult (T)"]},
    {"id": "Q3", "text": "Cât de dificil este pentru tine să respiri?", "options": ["Deloc (T)", "Foarte puțin (P)", "Moderat (P)", "Semnificativ (T)", "Foarte greu (T)"]},
    {"id": "Q4", "text": "Cât de mult durează un episod de tuse?", "options": ["Sub 10 secunde (P)", "10-30 sec (P)", "30 sec-2 min (P)", "2-4 min (T)", ">4 min (T)"]},
    {"id": "Q5", "text": "Cât de frecvent îți vine să tușești?", "options": ["Foarte rar (P)", "Ocazional (P)", "Moderat (P)", "Frecvent (T)", "Foarte frecvent (T)"]},
    {"id": "Q6", "text": "Cât de puternic simți durerea în piept?", "options": ["Deloc (P)", "Ușor (P)", "Moderat (P)", "Intens (P)", "Foarte intens (P)"]},
    {"id": "Q7", "text": "Cât de productivă este tusea ta?", "options": ["Deloc (T)", "Foarte puțin (P)", "Moderată (P)", "Multă (P)", "Foarte multă (P)"]},
    {"id": "Q8", "text": "Cum resimți frisoanele?", "options": ["Deloc", "Ușor (P)", "Moderat (P)", "Puternic (P)", "Foarte puternic (T)"]},
    {"id": "Q9", "text": "Cât de des ai dureri de cap?", "options": ["Niciodată (T)", "Rareori (P)", "Uneori (P)", "Des (P)", "Foarte des (T)"]},
    {"id": "Q10", "text": "Cât de intensă este durerea ta musculară?", "options": ["Deloc (P)", "Ușoară (P)", "Moderată (P)", "Puternică (P)", "Foarte puternică (P)"]},
    {"id": "Q11", "text": "Cât de des transpiri în timpul nopții?", "options": ["Niciodată", "Foarte rar (P)", "Ocazional (P)", "Frecvent (T)", "Permanent (T)"]},
    {"id": "Q12", "text": "Cât de mult te incomodează să respiri întins pe spate?", "options": ["Deloc (P)", "Foarte puțin (P)", "Moderat (P)", "Mult (T)", "Foarte Mult (T)"]},
    {"id": "Q13", "text": "Cât de des ai greață și/sau dureri abdominale?", "options": ["Niciodată (T)", "Rareori (P)", "Ocazional (P)", "Frecvent (P)", "Foarte frecvent (P)"]},
    {"id": "Q14", "text": "Cât de pronunțată este pierderea gustului/mirosului?", "options": ["Deloc (T)", "Foarte ușoară (P)", "Moderată (P)", "Pronunțată (P)", "Foarte pronunțată (P)"]},
    {"id": "Q15", "text": "Câte kg ai pierdut în ultimele 3 luni?", "options": ["Niciun kg (P)", "1–2 kg (P)", "3–5 kg (P)", "6–10 kg (T)", ">10 kg (T)"]},
    {"id": "Q16", "text": "Câte episoade de tuse au fost cu sânge?", "options": ["Niciunul (P)", "Foarte puține (P)", "Puține (P)", "Multe (T)", "Foarte multe (T)"]},
    {"id": "Q17", "text": "Cât de mult efort depui la respirație?", "options": ["Deloc (P)", "Foarte puțin (P)", "Moderat (P)", "Mult (T)", "Foarte mult (T)"]},
    {"id": "Q18", "text": "Cât de des ai avut ganglionii gâtului inflamați?", "options": ["Niciodată (T)", "Foarte rar (P)", "Ocazional (P)", "Frecvent (P)", "Permanent (P)"]},
    {"id": "Q19", "text": "Cât de mult ți s-a redus pofta de mâncare?", "options": ["Deloc (P)", "Foarte puțin (P)", "Moderată (P)", "Foarte mult (T)", "Nu mai mănânc (T)"]},
    {"id": "Q20", "text": "Cât de des ai avut febră intermitentă?", "options": ["Niciodată", "Foarte rar (P)", "Ocazional (P)", "Des (P)", "Foarte Des (T)"]}
]

# --- 4. FORMULAR UI ---
st.markdown("<h2 style='text-align: center;'>Chestionar Simptomatologie</h2>", unsafe_allow_html=True)
with st.form("main_form"):
    raw_indices = []
    col1, col2 = st.columns(2)
    for i, q in enumerate(questions):
        with (col1 if i < 10 else col2):
            choice = st.selectbox(q['text'], q['options'], key=q['id'])
            raw_indices.append(q['options'].index(choice))
    
    submit = st.form_submit_button("ANALIZEAZĂ CAZUL", use_container_width=True)

# --- 5. LOGICA DE INTERPRETARE (Aici este noutatea) ---
if submit:
    # A. Normalizare date
    input_numeric = np.array(raw_indices).astype(float) / 4.0
    
    # B. Predicție
    prediction = model.predict(input_numeric.reshape(1, -1), verbose=0)[0][0]
    
    # C. EXTRAGERE PONDERI (Ceea ce a învățat RN)
    # weights[0] sunt ponderile celor 20 de intrări
    weights = model.layers[0].get_weights()[0].flatten()
    
    # Influența = Ponderea * Valoarea introdusă
    influence = weights * input_numeric
    
    # Pregătire date pentru explicație
    expl_data = []
    for i in range(20):
        expl_data.append({"Simptom": questions[i]['text'], "Scor": influence[i]})
    
    # Sortăm după impact (cele mai mari valori absolute)
    expl_data.sort(key=lambda x: abs(x['Scor']), reverse=True)

    # D. AFIȘARE REZULTAT
    st.divider()
    res_c1, res_c2 = st.columns(2)
    
    if prediction >= 0.5:
        res_c1.error(f"### DIAGNOSTIC: TUBERCULOZĂ (T)")
        top_motive = [x for x in expl_data if x['Scor'] > 0][:3]
    else:
        res_c1.success(f"### DIAGNOSTIC: PNEUMONIE (P)")
        top_motive = [x for x in expl_data if x['Scor'] < 0][:3]

    res_c2.metric("Încredere Model", f"{prediction*100 if prediction >= 0.5 else (1-prediction)*100:.2f}%")

    # E. EXPLICAȚIA DECIZIEI
    st.subheader("🔍 De ce acest diagnostic?")
    st.write("Modelul a fost influențat cel mai mult de:")
    
    m_cols = st.columns(3)
    for idx, mot in enumerate(top_motive):
        with m_cols[idx]:
            st.info(f"**{mot['Simptom']}**")

    # F. GRAFIC DETALIAT (Toate cele 20 de influențe)
    with st.expander("Vezi analiza matematică a tuturor simptomelor"):
        expl_data.sort(key=lambda x: x['Scor'])
        fig = go.Figure(go.Bar(
            x=[x['Scor'] for x in expl_data],
            y=[x['Simptom'] for x in expl_data],
            orientation='h',
            marker_color=['red' if x['Scor'] > 0 else 'blue' for x in expl_data]
        ))
        fig.update_layout(title="Contribuția simptomelor (Roșu -> T | Albastru -> P)", height=600)
        st.plotly_chart(fig, use_container_width=True)