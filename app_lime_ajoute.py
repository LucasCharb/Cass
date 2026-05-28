import streamlit as st
import joblib
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular

# Config page
st.set_page_config(
    page_title="AI4FA",
    page_icon="",
    layout="wide"
)

# Importer modèle
@st.cache_resource
def load_model():
    data = joblib.load("random_forest(3).pkl")
    return data["model"], data["seuil"], data["ranking"]

model, seuil, ranking = load_model()

# Données synthétiques pour LIME
@st.cache_resource
def build_lime_explainer(_model):
    np.random.seed(42)
    n = 500

    binary_cols = [
        'Palpitations', 'Sexe', 'Tabac', 'HTA', 'Hypercholesterolémie',
        'DNID', 'OH', 'Flutter', 'Rao', 'RM', 'IM', 'BPCO',
        'Cardiopathie ischémique', 'IDM', 'SAS', 'artérite', 'IC',
        'Hyperthiroidie', 'HVG/CMH', 'Anévrisme Ao', 'ESA', 'AVC/AIT',
        'Athérome carotidien', 'Atcd fam - DNID', 'Atcd fam - HTA',
        'Atcd fam - AVC', 'Atcd fam - troubles rythme ', 'Atcd fam - FA',
        'Atcd fam - IDM'
    ]

    continuous_ranges = {
        'Age':          (30, 90),
        'Poids(kg)':    (45, 130),
        'Taille(cm)':   (150, 195),
        'IMC':          (16, 45),
        'FC (batt/min)':(40, 120),
        'P (ms)':       (60, 160),
        'PR (ms)':      (100, 280),
        'QRS (ms)':     (60, 160),
        'QT (ms)':      (300, 550),
        'QTc (ms)':     (350, 550),
        'QRS (°)':      (-90, 180),
    }

    data_synth = {}

    for col in FEATURE_NAMES:
        if col in binary_cols:
            data_synth[col] = np.random.randint(0, 2, n).astype(float)
        elif col in continuous_ranges:
            lo, hi = continuous_ranges[col]
            data_synth[col] = np.random.uniform(lo, hi, n)
        else:
            data_synth[col] = np.random.uniform(0, 1, n)

    X_synth = pd.DataFrame(data_synth, columns=FEATURE_NAMES)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_synth.values,
        feature_names=FEATURE_NAMES,
        class_names=['Sain', 'Malade'],
        mode='classification',
        discretize_continuous=True
    )
    return explainer

#Constantes des features
FEATURE_NAMES = [
    'Palpitations', 'Age', 'Sexe', 'Poids(kg)', 'Taille(cm)', 'IMC', 'Tabac', 'HTA',
    'Hypercholesterolémie', 'DNID', 'OH', 'Flutter', 'Rao', 'RM', 'IM', 'BPCO',
    'Cardiopathie ischémique', 'IDM', 'SAS', 'artérite', 'IC', 'Hyperthiroidie',
    'HVG/CMH', 'Anévrisme Ao', 'ESA', 'AVC/AIT', 'Athérome carotidien', 'Atcd fam - DNID',
    'Atcd fam - HTA', 'Atcd fam - AVC', 'Atcd fam - troubles rythme ', 'Atcd fam - FA',
    'Atcd fam - IDM', 'FC (batt/min)', 'P (ms)', 'PR (ms)', 'QRS (ms)', 'QT (ms)',
    'QTc (ms)', 'QRS (°)',
]

#Interface
st.title("AI4FA")
st.markdown("Renseignez les informations du patient pour générer le rapport d'analyse.")

#Identité
c_prenom, c_nom, c_motif, c_rdv = st.columns(4)
with c_prenom:
    prenom = st.text_input("Prénom", value="Callie")
with c_nom:
    nom = st.text_input("Nom", value="Moreau")
with c_motif:
    motif_input = st.text_input("Motif", value="Palpitations")
with c_rdv:
    rdv_input = st.text_input("Rendez-vous", value="12/12/2025 – 14h00")

st.divider()

#Données biométriques et ECG
st.subheader("Données biométriques et ECG")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    p    = st.number_input("P (ms)",        min_value=0.0, value=None)
    pr   = st.number_input("PR (ms)",       min_value=0.0, value=None)
with c2:
    qtc  = st.number_input("QTc (ms)",      min_value=0.0, value=None)
    qt   = st.number_input("QT (ms)",       min_value=0.0, value=None)
with c3:
    qrs_ms  = st.number_input("QRS (ms)",   min_value=0.0, value=None)
    qrs_deg = st.number_input("QRS (°)",    min_value=-180.0, max_value=180.0, value=None)
with c4:
    age  = st.number_input("Âge (ans)",     min_value=0, max_value=120, value=None, step=1)
    imc  = st.number_input("IMC",           min_value=0.0, value=None)
with c5:
    poids = st.number_input("Poids (kg)",   min_value=0.0, value=None)
    taille= st.number_input("Taille (cm)",  min_value=0.0, value=None)

c6, c7, c8, c9, c10 = st.columns(5)
with c6:
    fc   = st.number_input("FC (batt/min)", min_value=0.0, value=None)
with c7:
    sexe = st.selectbox("Sexe", options=[None, "Homme", "Femme"],
                        format_func=lambda x: "Sélectionner..." if x is None else x)

st.divider()


#Antécédents
st.subheader("Antécédents et facteurs de risque")

def oui_non(label):
    return st.selectbox(label, options=[None, "Non", "Oui"],
                        format_func=lambda x: "—" if x is None else x,
                        key=label)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Antécédents médicaux/cardiovasculaires**")
    ic               = oui_non("IC")
    avc_ait          = oui_non("AVC/AIT")
    atherome         = oui_non("Athérome carotidien")
    cardiopathie     = oui_non("Cardiopathie ischémique")
    anevrisme        = oui_non("Anévrisme Ao")
    arterite         = oui_non("artérite")
    rm               = oui_non("RM")
    im               = oui_non("IM")
    esa              = oui_non("ESA")
    hvg_cmh          = oui_non("HVG/CMH")
    idm              = oui_non("IDM")
    bpco             = oui_non("BPCO")
    sas              = oui_non("SAS")
    rao              = oui_non("Rao")
    hyperthyroidie   = oui_non("Hyperthiroidie")
    flutter          = oui_non("Flutter")
    palpitations     = oui_non("Palpitations")

with col2:
    st.markdown("**Antécédents familiaux**")
    atcd_fa          = oui_non("Atcd fam - FA")
    atcd_avc         = oui_non("Atcd fam - AVC")
    atcd_hta         = oui_non("Atcd fam - HTA")
    atcd_idm         = oui_non("Atcd fam - IDM")
    atcd_dnid        = oui_non("Atcd fam - DNID")
    atcd_rythme      = oui_non("Atcd fam - troubles rythme")

with col3:
    st.markdown("**Facteurs de risque**")
    hypercholesterol = oui_non("Hypercholestérolémie")
    tabac            = oui_non("Tabac")
    dnid             = oui_non("DNID")
    oh               = oui_non("OH")
    hta              = oui_non("HTA")

st.divider()


#Encodage pour la prédiction
def encode_bin(val):
    if val is None:
        return None
    if val == "Homme":
        return 0
    if val == "Femme":
        return 1
    return 1 if val == "Oui" else 0

valeurs_dict = {
    'Palpitations':               encode_bin(palpitations),
    'Age':                        age,
    'Sexe':                       encode_bin(sexe),
    'Poids(kg)':                  poids,
    'Taille(cm)':                 taille,
    'IMC':                        imc,
    'Tabac':                      encode_bin(tabac),
    'HTA':                        encode_bin(hta),
    'Hypercholesterolémie':       encode_bin(hypercholesterol),
    'DNID':                       encode_bin(dnid),
    'OH':                         encode_bin(oh),
    'Flutter':                    encode_bin(flutter),
    'Rao':                        encode_bin(rao),
    'RM':                         encode_bin(rm),
    'IM':                         encode_bin(im),
    'BPCO':                       encode_bin(bpco),
    'Cardiopathie ischémique':    encode_bin(cardiopathie),
    'IDM':                        encode_bin(idm),
    'SAS':                        encode_bin(sas),
    'artérite':                   encode_bin(arterite),
    'IC':                         encode_bin(ic),
    'Hyperthiroidie':             encode_bin(hyperthyroidie),
    'HVG/CMH':                    encode_bin(hvg_cmh),
    'Anévrisme Ao':               encode_bin(anevrisme),
    'ESA':                        encode_bin(esa),
    'AVC/AIT':                    encode_bin(avc_ait),
    'Athérome carotidien':        encode_bin(atherome),
    'Atcd fam - DNID':            encode_bin(atcd_dnid),
    'Atcd fam - HTA':             encode_bin(atcd_hta),
    'Atcd fam - AVC':             encode_bin(atcd_avc),
    'Atcd fam - troubles rythme ': encode_bin(atcd_rythme),
    'Atcd fam - FA':              encode_bin(atcd_fa),
    'Atcd fam - IDM':             encode_bin(atcd_idm),
    'FC (batt/min)':              fc,
    'P (ms)':                     p,
    'PR (ms)':                    pr,
    'QRS (ms)':                   qrs_ms,
    'QT (ms)':                    qt,
    'QTc (ms)':                   qtc,
    'QRS (°)':                    qrs_deg,
}

champs_vides = any(v is None for v in valeurs_dict.values())

#Bouton prédire
if st.button("Lancer l'analyse", type="primary", use_container_width=True, disabled=champs_vides):
    
    #Prédiction
    X = pd.DataFrame([valeurs_dict], columns=FEATURE_NAMES).astype(float)
    probas = model.predict_proba(X)[0]
    label  = 1 if probas[1] >= seuil else 0
    score_malade_pct = probas[1] * 100
    
    #LIME
    with st.spinner("Génération du rapport et calcul de l'explicabilité en cours..."):
        lime_explainer = build_lime_explainer(model)
        
        #On calcule pour toutes les variables
        exp = lime_explainer.explain_instance(X.values[0], model.predict_proba, num_features=len(FEATURE_NAMES))
        lime_list = exp.as_list()
        
        #Séparation et tri
        facteurs_contributifs = sorted([f for f in lime_list if f[1] > 0], key=lambda x: x[1], reverse=True)
        facteurs_non_contributifs = sorted([f for f in lime_list if f[1] <= 0], key=lambda x: x[1]) # Les plus négatifs en premier

        top_10_contributifs = facteurs_contributifs[:10]
        top_10_protecteurs = facteurs_non_contributifs[:10]

    st.markdown("---")
    
    
    #Entete Patient
    with st.container(border=True):
        col_header1, col_header2, col_header3, col_header4 = st.columns([2, 2, 2, 2])
        
        with col_header1:
            st.markdown(f"### {prenom} {nom} <span style='font-size:18px; color:gray; font-weight:normal;'>— {int(age)} ans</span>", unsafe_allow_html=True)
            st.write(f"**Taille :** {taille} cm")
            st.write(f"**Poids :** {poids} kg")
            st.write(f"**IMC :** {imc}")
            
        with col_header2:
            st.write("") 
            st.write(f"**Motif :** {motif_input}")
            st.write(f"**Rendez-vous :** {rdv_input}")
            
        with col_header3:
            st.empty() 
            
        with col_header4:
            st.button("Télécharger ECG", use_container_width=True)
            st.button("Télécharger Échographie", use_container_width=True)
            st.button("Afficher historique patient", use_container_width=True)
            st.button("Exporter compte-rendu", use_container_width=True)

    #Score & Facteurs Contributifs
    with st.container(border=True):
        
        #La décision clinique
        if label == 1:
            st.markdown("<h2 style='text-align: center; color: #ef4444; margin-bottom: 0px;'>Risque élevé de FA</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: #10b981; margin-bottom: 0px;'>Risque peu élevé de FA</h2>", unsafe_allow_html=True)
            
        #Probabilité estimée
        st.markdown(f"<h4 style='text-align: center; color: #475569; margin-top: 5px;'>Probabilité estimée : {score_malade_pct:.1f} %</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 14px;'>Calculé en rythme sinusal à partir de données cliniques multimodales.</p>", unsafe_allow_html=True)
        
        st.write("") 
        
        #Message d'interprétation pour éviter la confusion sur les pourcentages
        st.info("Note d'interprétation : Les pourcentages affichés ci-dessous n'ont aucun lien additif direct avec le pourcentage global de risque (ils ne s'additionnent pas pour former le score final). Ils mesurent uniquement l'écart d'impact d'une caractéristique par rapport à un profil patient moyen de référence. Par exemple, un facteur à -11 % abaisse le risque du patient de 11 points par rapport à la moyenne de la population étudiée.")
        
        #10 facteurs contributifs principaux
        st.markdown("**Facteurs contributifs principaux**")
        if top_10_contributifs:
            pills_html = ""
            for f in top_10_contributifs:
                rule_text = f[0].replace("<=", "≤").replace(">", ">")
                weight = f[1]
                pct = weight * 100
                
                if pct >= 0.1:
                    pct_display = f"+{pct:.1f} %"
                else:
                    pct_display = "<0.1 %"
                
                display_text = f"{rule_text} ({pct_display})"
                pills_html += f"<span style='display: inline-block; background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 20px; padding: 6px 14px; margin: 4px; font-size: 14px; color: #334155;'>{display_text}</span>"
            
            st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
        else:
            st.info("Aucun facteur clinique n'augmente significativement le risque.")

    #Éléments non contributifs
    with st.container(border=True):
        st.markdown("**Éléments protecteurs ou neutres principaux**")
        if top_10_protecteurs:
            for f in top_10_protecteurs:
                rule_text = f[0].replace("<=", "≤").replace(">", ">")
                weight = f[1]
                pct_abs = abs(weight) * 100
                
                if weight <= -0.001:
                    if pct_abs >= 0.1:
                        pct_display = f"-{pct_abs:.1f} %"
                    else:
                        pct_display = "<-0.1 %"
                    st.markdown(f"- {rule_text} : **{pct_display}**")
                else:
                    st.markdown(f"- {rule_text} : **Neutre (0.0 %)**")
        else:
            st.write("- Aucun.")

    #Détail complet
    st.write("")
    with st.expander("Voir l'analyse détaillée des 40 variables cliniques"):
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("**Tous les facteurs aggravants**")
            if facteurs_contributifs:
                for f in facteurs_contributifs:
                    rule_text = f[0].replace("<=", "≤").replace(">", ">")
                    pct = f[1] * 100
                    
                    if pct >= 0.1:
                        pct_display = f"+{pct:.1f} %"
                    else:
                        pct_display = "<0.1 %"
                        
                    st.write(f"- {rule_text} : **{pct_display}**")
            else:
                st.write("Aucun")
                
        with col_exp2:
            st.markdown("**Tous les facteurs protecteurs / neutres**")
            if facteurs_non_contributifs:
                for f in facteurs_non_contributifs:
                    rule_text = f[0].replace("<=", "≤").replace(">", ">")
                    pct_abs = abs(f[1]) * 100
                    
                    if f[1] <= -0.001:
                        if pct_abs >= 0.1:
                            pct_display = f"-{pct_abs:.1f} %"
                        else:
                            pct_display = "<-0.1 %"
                    else:
                        pct_display = "Neutre (0.0 %)"
                        
                    st.write(f"- {rule_text} : **{pct_display}**")
            else:
                st.write("Aucun")