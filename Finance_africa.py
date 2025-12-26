# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %matplotlib inline

import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import streamlit as st 
import altair as alt
import streamlit_option_menu
from streamlit_option_menu import option_menu
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import warnings
import streamlit_jupyter 
import plotly.express as px
import streamlit_extras
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import joblib
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(
    page_title="Financial Inclusion",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")



st.markdown("""
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"],[data-testid="stToolbar"]  {
        background-color: #0b1a3b;
        color: #ffffff;
       
    }
    [data-testid="stExpander"] {
        background-color: #2f3f61;
        color : #ffffff
    }
    
     [data-testid="stSidebar"] {
        background-color: #2f3f61;
        color : #ffffff
    }
    
     [data-testid="stSelectbox"] {
    background-color: #0000;  /* Change la couleur de fond du selectbox */
    color: #000000;             /* Change la couleur du texte dans le selectbox */
    border: 2px solid #0000;  /* Ajoute une bordure autour du selectbox */
    font-size: 16px;            /* Change la taille de la police du texte */
    border-radius: 8px;         /* Change le rayon des coins pour avoir des bords arrondis */
    padding: 10px;              /* Modifie l'espacement interne (padding) du selectbox */
}
    [data-testid="stSelectbox"] option {
    background-color: #0b1a3b;  /* Change la couleur de fond des options dans le selectbox */
    color: #ffffff;             /* Change la couleur du texte des options */
    padding: 10px;              /* Ajoute un espacement à l'intérieur de chaque option */
}
    [data-testid="stSelectbox"]:hover {
    background-color: #8dc497;  /* Change la couleur de fond du selectbox au survol */
}
 /* Change la couleur de fond du Number input */
    [data-testid="stNumberInput"] {
    background-color: #0000;  /* Change la couleur de fond du selectbox */
    color: #000000;             /* Change la couleur du texte dans le selectbox */
    border: 2px solid #0000;  /* Ajoute une bordure autour du selectbox */
    font-size: 16px;            /* Change la taille de la police du texte */
    border-radius: 8px;         /* Change le rayon des coins pour avoir des bords arrondis */
    padding: 10px;              /* Modifie l'espacement interne (padding) du selectbox */
}
    [data-testid="stNumberInput"] option {
    background-color: #0b1a3b;  /* Change la couleur de fond des options dans le selectbox */
    color: #ffffff;             /* Change la couleur du texte des options */
    padding: 10px;              /* Ajoute un espacement à l'intérieur de chaque option */
}
    [data-testid="stNumberInput"]:hover {
    background-color: #8dc497;  /* Change la couleur de fond du selectbox au survol */
}

    </style>
    """, unsafe_allow_html = True)
#importation et nettoyage des données
df = pd.read_csv("Financial_inclusion_dataset.csv")
header = st.container()
sidebar = st.sidebar
#data = st.expander("🌍Les matchs", )
#figure = st.expander("Les figues")
footer = st.container()
#img = Image.open("ball.png")

# -

with sidebar:
   
      
    st.header("", divider="green")
    selected = option_menu("Menu", ['DATA','Machine Learning','Info'], 
    icons=['data','machine learning', 'rss','info'], menu_icon="cast", default_index=0,

    styles={
        "container": {
            "background-color": "#dbdeee",  # Couleur de fond du conteneur principal
            "padding": "10px",              # Espacement interne
            "border-radius": "10px",         # Bords arrondis
            "margin": "10px",
            "color": "#ffffff",
            "icon" :  "#ffffff"
        },
        "icon": {"color": "#054610", "font-size": "16px"},  # Style des icônes
        "nav-link": {
            "color": "#2f3f61",             # Couleur du texte des options
            "background-color": "#ffffff",  # Couleur de fond des options
            "border-radius": "5px",         # Bords arrondis pour chaque option
            "margin": "7px",                # Espacement entre les options
        },
        "nav-link-selected": {
            "background-color": "#8dc497",  # Couleur de fond de l'option sélectionnée
            "font-weight": "normal",          # Texte en gras pour l'option sélectionnée
        },
        "nav-link-hover": {
            "background-color": "#ad1c05",  # Couleur de fond au survol
            "color": "#ffffff",             # Couleur du texte au survol
        },
    }
                          )
    st.header("", divider="green")
    
    st.header("", divider="green")

if selected =="DATA":
    st.markdown("<h5 style='text-align: center; color:white;'>En tête dataframe</h5>", unsafe_allow_html=True)
    st.dataframe(df.head())
    
    #st.dataframe(df.info())
    
    #st.dataframe(df.isnull().sum()) 
    
    st.markdown("<h5 style='text-align: center; color:white;'>Description des données</h5>", unsafe_allow_html=True)
    st.dataframe(df.describe()) 
    
    #st.dataframe(df.duplicated().value_counts()) 
    
   
    profile = ProfileReport(
        df,
        title="Rapport de Profilage des Données",
        explorative=True
    )
    
    
    profile
    
    st.title("Profiling des Données")
    st_profile_report(profile)
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot de {col}")
            st.pyplot(plt)   # <- ici, on utilise st.pyplot
            plt.clf()        # <- nettoyer la figure pour le prochain plot
            


if selected =="Machine Learning":
    #importation et nettoyage des données
    df = pd.read_csv("Financial_inclusion_dataset.csv")

   
        
    # Variable cible
    y = df['bank_account']
    
    # Variables explicatives
    X = df.drop(columns=['bank_account', 'uniqueid'])  # uniqueid n'apporte rien au ML
    
    #Encodage    
    X = pd.get_dummies(X, drop_first=True)
    

    le = LabelEncoder()
    y = le.fit_transform(y)   # Yes/No → 1/0

    
    # +
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    st.title("Résultats du modèle de régression logistique")
    st.write("Accuracy :", accuracy_score(y_test, y_pred))
    st.write("\nRapport de classification :\n", classification_report(y_test, y_pred))
    
    
    # +
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    plt.title("Matrice de confusion")
    plt.show()
    st.pyplot(plt)   # <- ici, on utilise st.pyplot
    plt.clf()        # <- nettoyer la figure pour le prochain plot
            
    
    
    # +
    #Ramdom Forest
    
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    st.title("Résultats du modèle random forest")
    st.write("Accuracy RF :", accuracy_score(y_test, y_pred_rf))
    st.write(classification_report(y_test, y_pred_rf))
    
    # -
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances.sort_values(ascending=False).head(10)
    st.title("Top 10 des variables les plus importantes")
    st.write(importances.sort_values(ascending=False).head(10))
    joblib.dump(model, "model_bank_account.pkl")
    joblib.dump(X.columns.tolist(), "features.pkl")



    # Charger le modèle et les features
    model = joblib.load("model_bank_account.pkl")
    features = joblib.load("features.pkl")
    st.title("🏦 Prédiction de possession d'un compte bancaire")
    
    st.write("Veuillez remplir les informations suivantes :")
    
    # -------- Formulaire --------
    with st.form("bank_form"):
        pays = sorted(
        set(df["country"].unique()))
        country =st.selectbox("Choisissez un pays",pays,placeholder="Tapez un pays...",key="A")
        ans = sorted(
        set(df["year"].unique()))
        year =st.selectbox("Choisissez une anné",ans,placeholder="l'année...",key="B")
        #year = st.number_input("Année", 2016, 2025, 2018)
        location_type = st.selectbox("Type de localisation", ["Urban", "Rural"])
        cellphone_access = st.selectbox("Accès au téléphone", ["Yes", "No"])
        household_size = st.number_input("Taille du ménage", 1, 20, 4)
        age = st.number_input("Âge", 18, 100, 30)
        gender = st.selectbox("Genre", ["Male", "Female"])
        relationship = st.selectbox(
            "Relation avec le chef de ménage",
            ["Head of Household", "Spouse", "Child", "Other"]
        )
        marital_status = st.selectbox(
            "Statut matrimonial",
            ["Married", "Single", "Divorced", "Widowed"]
        )
        education = st.selectbox(
            "Niveau d'éducation",
            ["No formal education", "Primary education",
             "Secondary education", "Tertiary education"]
        )
        job = st.selectbox(
            "Type d'emploi",
            ["Farming", "Self employed", "Formally employed", "Informally employed"]
        )
    
        submitted = st.form_submit_button("🔍 Prédire")

# -------- Prédiction --------
        if submitted:
            input_data = pd.DataFrame([{
                "country": country,
                "year": year,
                "location_type": location_type,
                "cellphone_access": cellphone_access,
                "household_size": household_size,
                "age_of_respondent": age,
                "gender_of_respondent": gender,
                "relationship_with_head": relationship,
                "marital_status": marital_status,
                "education_level": education,
                "job_type": job
            }])
        
            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=features, fill_value=0)
        
            prediction = model.predict(input_encoded)[0]
        
            if prediction == 1:
                st.success("✅ La personne a un compte bancaire")
            else:
                st.error("❌ La personne n'a PAS de compte bancaire")
if selected =="Info":
        "Description du jeu de données : Ce jeu de données contient des informations démographiques et relatives aux services financiers utilisés par environ 33 600 personnes en Afrique de l’Est. Le modèle d’apprentissage automatique a pour objectif de prédire quelles personnes sont les plus susceptibles de posséder ou d’utiliser un compte bancaire."




