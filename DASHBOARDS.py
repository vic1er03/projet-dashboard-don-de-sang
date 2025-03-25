import streamlit as st
import pandas as pd
import plotly.express as px
import folium
import geopandas as gpd
#from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Charger les données depuis le fichier local
@st.cache_data
def load_data():
    file_path = r"C:\Users\Ultra Tech\Desktop\Challenge dataset fin.xlsx"  # Chemin du fichier dans Colab
    xls = pd.ExcelFile(file_path)

    df_2019 = pd.read_excel(xls, sheet_name="2019")
    df_2020 = pd.read_excel(xls, sheet_name="2020")
    df_volontaires = pd.read_excel(xls, sheet_name="Volontaire")
    df_condition = pd.read_excel(xls, sheet_name="Feuil1")
    return df_2019, df_2020, df_volontaires, df_condition

df_2019, df_2020, df_volontaires, df_condition = load_data()

# Fusionner les données des années 2019 et 2020
df = pd.concat([df_2019, df_2020], ignore_index=True)

# Nettoyage et transformation des données
df.rename(columns={"Arrondissement de résidence": "Arrondissement",
                   "Quartier de Résidence": "Quartier",
                   "Genre": "Sexe",
                   "Profession": "Métier"}, inplace=True)

df["Date de naissance"] = pd.to_datetime(df["Date de naissance"], errors="coerce")
df["Âge"] = 2025 - df["Date de naissance"].dt.year

#st.set_page_config(page_title="Dashboard--Campagne de don de sang",layout="wide")
#creation de colonne
col1, col2, col3, col4, col5, col6, col7 =st.columns(7)
#Menu de navigation
menu=["Accueil", "Carte de distribution des donneurs", "Conditions de santé et Eligibilité", "Profilage des Donneurs", "Fidélisation des Donneurs", "Analyse de Sentiment", "Prédiction d'Éligibilité"]
choix=st.radio("Navigation", menu, horizontal= True)

if choix== "Accueil":  
    st.title("📊 Tableau de Bord - Campagne de Don de Sang")
    st.image(r"C:\Users\Ultra Tech\Downloads\don de sang.jpeg")
    st.write("Bienvenue sur le tableau de bord interactif!")
    


elif choix=="Carte de distribution des donneurs":
    st.title("Carte de distribution des donneurs")
    fig=px.scatter_mapbox(df_volontaires,lat="latitude", lon="longitude", zoom=6, height=600, mapbox_style="carto-positron", title= "Repartition des Individus par Zone")
    
    st.sidebar.header("Filtres")
    arrondissement = st.sidebar.selectbox("Sélectionner un arrondissement", df_volontaires["Arrondissement_de_résidence_"].dropna().unique())
    
    df_filtered = df_volontaires[df_volontaires["Arrondissement_de_résidence_"] == arrondissement]
    
    st.plotly_chart(fig)
    # Distribution des donneurs
    st.subheader("📊 Distribution des Donneurs")
    fig = px.histogram(df_filtered, x="Age", nbins=20, title="Répartition des âges")
    st.plotly_chart(fig)
    
    #fig1 = px.pie(df_filtered, names="Sexe", title="Répartition par Genre")
    #st.plotly_chart(fig1)
    fig = px.bar(df_filtered["Profession_"].value_counts().head(10), title="Top 10 des professions")
    st.plotly_chart(fig)

# Conditions de santé

elif choix=="Conditions de santé et Eligibilité":

    st.title("🏥 Conditions de Santé et Éligibilité")
    """columns_sante = ["Raison de non-eligibilité totale  [Hypertendus]",
                     "Raison de non-eligibilité totale  [Diabétique]",
                     "Raison de non-eligibilité totale  [Asthmatiques]",
                     "Raison de non-eligibilité totale  [Cardiaque]"]
    eligibilite_counts = df_filtered[columns_sante].notna().sum()
    fig = px.bar(eligibilite_counts, title="Facteurs de non-éligibilité les plus courants")
    st.plotly_chart(fig)"""
    Eligibilité = st.sidebar.selectbox("Sélectionner un état  d'eligibilité", df_condition["ÉLIGIBILITÉ_AU_DON."].dropna().unique())
    
    df_filtered = df_condition[df_condition["ÉLIGIBILITÉ_AU_DON."] == Eligibilité]
    
    fig = px.pie(df_filtered["Condition de Santé"], title="Repartition des condition de santé selon l'état d'éligibilité")
    st.plotly_chart(fig)
    profilas = st.sidebar.selectbox("Sélectionner un critère", df_volontaires.columns)
    df_filtered = df_volontaires[profilas]
    
    fig = px.bar(df_filtered, title=f"Repartition de la distribution des donneurs selon le {profilas}")
    st.plotly_chart(fig)

    # Sélection de la colonne d'éligibilité
    elig_col = st.selectbox("Sélectionnez la colonne d'éligibilité", ["ÉLIGIBILITÉ AU DON."])

    # Sélection des colonnes des conditions de santé (exclut la colonne d'éligibilité)
    health_cols = st.multiselect("Sélectionnez les colonnes des conditions de santé", 
                                 [col for col in df_volontaires.columns if col != elig_col])

    if health_cols:
        # Transformation des données pour un format adapté à un diagramme en barres
        data_list = []

        for col in health_cols:
            count = df_volontaires.groupby([col, elig_col]).size().reset_index(name="count")
            count["Condition"] = col
            data_list.append(count)

        df_melted = pd.concat(data_list)

        # Création du graphique en barres avec Plotly
        fig = px.bar(df_melted, 
                     x="Condition", 
                     y="count", 
                     color=elig_col, 
                     barmode="group",
                     facet_col=df_melted.columns[0],  # "Oui" / "Non"
                     labels={"count": "Nombre de personnes", elig_col: "Éligibilité"},
                     title="Nombre de personnes éligibles et non éligibles selon les conditions de santé")

        # Affichage du graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)
    

# Profilage des Donneurs
elif choix=="Profilage des Donneurs":
    st.title("🔍 Profilage des Donneurs")
    df_volontaires=df_volontaire[df_volontaires["ÉLIGIBILITÉ_AU_DON."]=="Eligible"]
    df_cluster = df_volontaires[["Age","Genre_","Niveau_d'etude","Profession_","Situation_Matrimoniale_(SM)"]].dropna()
    kmeans = KMeans(n_clusters=3)
    df_cluster["Cluster"] = kmeans.fit_predict(df_cluster)
    fig = px.scatter(df_cluster, x="Age", y=df_cluster["Cluster"], color=df_cluster["Cluster"].astype(str),
                     title="Segmentation des donneurs par âge")
    st.plotly_chart(fig)
    fig1 = px.scatter(df_cluster, x="Genre_", y=df_cluster["Cluster"], color=df_cluster["Cluster"].astype(str),
                     title="Segmentation des donneurs par genre")
    st.plotly_chart(fig1)
    fig2 = px.scatter(df_cluster, x="Niveau_d'etude", y=df_cluster["Cluster"], color=df_cluster["Cluster"].astype(str),
                     title="Segmentation des donneurs par Niveau d'etude")
    st.plotly_chart(fig2)

    profila = st.sidebar.selectbox("Sélectionner un critère", df_volontaires.columns)
    df_filtered = df_volontaires[profila]
    
    fig = px.bar(df_filtered, title=f"Repartition de la distribution des donneurs selon le {profila}")
    st.plotly_chart(fig)
    

# Fidélisation des Donneurs
elif choix=="Fidélisation des Donneurs":
    st.title("🔄 Fidélisation des Donneurs")
    if "Date de remplissage de la fiche" in df.columns:
        df["Date de remplissage de la fiche"] = pd.to_datetime(df["Date de remplissage de la fiche"], errors="coerce")
        df["Année"] = df["Date de remplissage de la fiche"].dt.year
        fidelisation = df["Année"].value_counts().sort_index()
        fig = px.line(fidelisation, title="Évolution des dons au fil des années")
        st.plotly_chart(fig)

# Analyse de Sentiment
elif choix=="Analyse de Sentiment":
    st.subheader("📝 Analyse de Sentiment")
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    if "Si autres raison préciser" in df.columns:
        df["Sentiment"] = df["Si autres raison préciser"].dropna().apply(lambda x: sia.polarity_scores(str(x))["compound"])
        sentiment_counts = df["Sentiment"].apply(lambda x: "Positif" if x > 0 else "Négatif" if x < 0 else "Neutre").value_counts()
        fig = px.pie(sentiment_counts, names=sentiment_counts.index, title="Répartition des sentiments")
        st.plotly_chart(fig)
        text = " ".join(str(f) for f in df["Si autres raison préciser"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array(), caption="Nuage de Mots des Feedbacks", use_column_width=True)

# Prédiction d'Éligibilité
elif choix=="Prédiction d'Éligibilité":
    st.title("🔮 Prédiction d'Éligibilité (à implémenter)")
    st.write("👨‍💻 **Prochaines étapes :** Entraîner un modèle et l'intégrer ici !")