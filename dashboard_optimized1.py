"""Tableau de bord interactif pour l'analyse des données de donneurs de sang.
Ce script crée un tableau de bord Streamlit avec des visualisations innovantes
pour répondre aux objectifs du concours de data visualisation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.metrics import roc_curve, auc, classification_report,confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
# Créer un pipeline pour le prétraitement et la modélisation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import base64
import warnings
import io
warnings.filterwarnings('ignore')

def paginate_dataframe(dataframe, page_size=10):
    """Ajoute une pagination à un DataFrame pour améliorer les performances."""
    # Obtenir le nombre total de pages
    n_pages = len(dataframe) // page_size + (1 if len(dataframe) % page_size > 0 else 0)
    
    # Ajouter un sélecteur de page
    page = st.selectbox('Page', range(1, n_pages + 1), 1)
    
    # Afficher la page sélectionnée
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(dataframe))
    
    return dataframe.iloc[start_idx:end_idx]


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des Donneurs de Sang",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Définir les chemins des fichiers
data_2019_path = "data_2019_preprocessed.csv"
data_volontaire_path = "data_volontaire_preprocessed.csv"
#analysis_results_dir = r"C:\Users\Ultra Tech\Desktop\analysis_results"
#model_path = r"C:\Users\Ultra Tech\Desktop\preprocessor_eligibility.pkl"

# Fonction pour charger les données
@st.cache_data
@st.cache_data(ttl=3600, max_entries=2)

def load_data():
    
    #Charge les données prétraitées à partir des fichiers CSV.
    
    patr="Challenge_dataset_traité.csv"
    df_2019 = pd.read_csv(data_2019_path)
    df_volontaire = pd.read_csv(data_volontaire_path)
    df=pd.read_csv(patr)
    df_volontaires=pd.read_csv(patr)
    
    # Convertir les colonnes de dates au format datetime
    date_columns = [col for col in df_2019.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in df_2019.columns:
            try:
                df_2019[col] = pd.to_datetime(df_2019[col], errors='coerce')
            except:
                pass
    
    date_columns = [col for col in df_volontaire.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in df_volontaire.columns:
            try:
                df_volontaire[col] = pd.to_datetime(df_volontaire[col], errors='coerce')
            except:
                pass
    
    return df_2019, df_volontaire,df,df_volontaires

# Fonction pour créer une carte de distribution géographique
@st.cache_data
def create_geo_map(df, location_column, color_column=None, zoom_start=10):
    
    #Crée une carte interactive montrant la distribution géographique des donneurs.
    
    # Coordonnées approximatives pour Douala, Cameroun
    douala_coords = [4.0511, 9.7679]
    
    # Créer une carte centrée sur Douala
    m = folium.Map(location=douala_coords, zoom_start=zoom_start, tiles="OpenStreetMap")
    
    # Créer un dictionnaire de coordonnées pour les arrondissements de Douala
    # Ces coordonnées sont approximatives et devraient être remplacées par des données plus précises
    arrondissement_coords = {
        "Douala 1": [4.0511, 9.7679],
        "Douala 2": [4.0611, 9.7579],
        "Douala 3": [4.0711, 9.7479],
        "Douala 4": [4.0811, 9.7379],
        "Douala 5": [4.0911, 9.7279],
        "Douala 6": [4.1011, 9.7179],
        "Douala (Non précisé )": [4.0511, 9.7679],
        "Yaounde": [3.8480, 11.5021],
        "Edea": [3.8028, 10.1319]
    }
    
    # Ajouter un léger décalage aléatoire pour éviter la superposition des marqueurs
    np.random.seed(42)
    jitter = 0.02
    
    # Compter le nombre de donneurs par arrondissement
    location_counts = df[location_column].value_counts()
    
    # Créer un cluster de marqueurs pour une meilleure visualisation
    marker_cluster = MarkerCluster().add_to(m)
    
    # Données pour la carte de chaleur
    heat_data = []
    
    # Ajouter des marqueurs pour chaque arrondissement
    for arrondissement, count in location_counts.items():
        if arrondissement in arrondissement_coords:
            base_coords = arrondissement_coords[arrondissement]
            
            # Ajouter des points à la carte de chaleur
            for _ in range(int(count/10) + 1):  # Réduire le nombre de points pour la performance
                jittered_coords = [
                    base_coords[0] + np.random.uniform(-jitter, jitter),
                    base_coords[1] + np.random.uniform(-jitter, jitter)
                ]
                heat_data.append(jittered_coords)
            
            # Déterminer la couleur en fonction de la colonne de couleur si spécifiée
            if color_column and color_column in df.columns:
                # Filtrer le DataFrame pour cet arrondissement
                df_filtered = df[df[location_column] == arrondissement]
                
                # Compter les occurrences de chaque valeur de la colonne de couleur
                color_counts = df_filtered[color_column].value_counts()
                
                # Créer un popup avec ces informations
                popup_text = f"<b>{arrondissement}</b><br>Total: {count} donneurs<br><br>"
                for color_val, color_count in color_counts.items():
                    popup_text += f"{color_val}: {color_count} ({color_count/count*100:.1f}%)<br>"
                
                folium.Marker(
                    location=base_coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(icon="info-sign"),
                ).add_to(marker_cluster)
            else:
                # Popup simple avec juste le nombre de donneurs
                folium.Marker(
                    location=base_coords,
                    popup=f"<b>{arrondissement}</b><br>Nombre de donneurs: {count}",
                    icon=folium.Icon(icon="info-sign"),
                ).add_to(marker_cluster)
    
    # Ajouter une carte de chaleur
    HeatMap(heat_data, radius=15).add_to(m)
    
    return m

# Fonction pour créer un graphique de santé et éligibilité
@st.cache_data
def create_health_eligibility_chart(df):
    #Crée un graphique interactif montrant l'impact des conditions de santé sur l'éligibilité au don.
    
    # Identifier les colonnes de conditions de santé
    health_columns = [col for col in df.columns if any(term in col for term in 
                     ['Porteur', 'Opéré', 'Drepanocytaire', 'Diabétique', 'Hypertendus', 
                      'Asthmatiques', 'Cardiaque', 'Tatoué', 'Scarifié'])]
    
    # Créer un DataFrame pour stocker les résultats
    results = []
    
    for col in health_columns:
        # Créer une table de contingence
        contingency = pd.crosstab(df['ÉLIGIBILITÉ_AU_DON.'], df[col])
        
        # Calculer les pourcentages
        contingency_pct = contingency.div(contingency.sum(axis=0), axis=1) * 100
        
        # Extraire les données pour "Oui"
        if 'Oui' in contingency_pct.columns:
            for eligibility, percentage in contingency_pct['Oui'].items():
                results.append({
                    'Condition': col.split('[')[-1].split(']')[0],
                    'Éligibilité': eligibility,
                    'Pourcentage': percentage,
                    'Nombre': contingency.loc[eligibility, 'Oui']
                })
    
    # Créer un DataFrame à partir des résultats
    results_df = pd.DataFrame(results)
    
    # Créer un graphique à barres groupées avec Plotly
    fig = px.bar(
        results_df,
        x='Condition',
        y='Pourcentage',
        color='Éligibilité',
        barmode='group',
        text='Nombre',
        title="Impact des conditions de santé sur l'éligibilité au don",
        labels={'Pourcentage': 'Pourcentage de donneurs (%)', 'Condition': 'Condition de santé'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Personnaliser le graphique
    fig.update_layout(
        xaxis_title="Condition de santé",
        yaxis_title="Pourcentage de donneurs (%)",
        legend_title="Éligibilité",
        font=dict(size=12),
        height=600
    )
    
    return fig

# Fonction pour créer un graphique de clustering des donneurs
@st.cache_data
def create_donor_clustering(df):
    
    #Crée une visualisation interactive des clusters de donneurs.
    
    # Sélectionner les variables numériques pour le clustering
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Afficher les informations sur les colonnes numériques pour le débogage
    print(f"Colonnes numériques disponibles: {numeric_df.columns.tolist()}")
    print(f"Nombre de colonnes numériques: {numeric_df.shape[1]}")
    
    # Sélectionner uniquement les colonnes sans valeurs manquantes ou avec peu de valeurs manquantes
    # pour éviter les problèmes de dimensionnalité
    threshold = 0.5  # Colonnes avec moins de 50% de valeurs manquantes
    numeric_df = numeric_df.loc[:, numeric_df.isnull().mean() < threshold]
    print(f"Colonnes numériques après filtrage: {numeric_df.columns.tolist()}")
    
    if numeric_df.shape[1] >= 2:
        # Gérer les valeurs manquantes avant le clustering
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        
        # Imputer les valeurs manquantes
        imputed_values = imputer.fit_transform(numeric_df)
        
        # Créer un nouveau DataFrame avec les valeurs imputées
        numeric_df_imputed = pd.DataFrame(
            imputed_values,
            columns=numeric_df.columns,
            index=numeric_df.index
        )
        
        # Standardiser les données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df_imputed)
        
        # Appliquer K-means avec 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Ajouter les clusters au DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = clusters
        
        # Appliquer PCA pour la visualisation
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Créer un DataFrame pour la visualisation
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': clusters.astype(str)
        })
        
        # Ajouter des informations supplémentaires si disponibles
        if 'Age' in df.columns:
            pca_df['Age'] = df['Age'].values
        if 'Genre_' in df.columns:
            pca_df['Genre'] = df['Genre_'].values
        if 'Niveau_d\'etude' in df.columns:
            pca_df['Niveau_d\'études'] = df['Niveau_d\'etude'].values
        
        # Créer un graphique interactif avec Plotly
        fig = px.scatter(pca_df,
            render_mode='auto', 
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=pca_df.columns,
            title="Clustering des donneurs (PCA)",
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} de variance expliquée)',
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} de variance expliquée)'},
            color_discrete_sequence=px.colors.qualitative.Bold)
        
        # Personnaliser le graphique
        fig.update_layout(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} de variance expliquée)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} de variance expliquée)',
            legend_title="Cluster",
            font=dict(size=12),
            height=600)
        
        # Analyser les caractéristiques de chaque cluster
        cluster_stats = df_with_clusters.groupby('Cluster').agg({
            col: ['mean', 'std'] for col in numeric_df.columns
        })
        
        return fig, cluster_stats, df_with_clusters
    else:
        return None, None, df

# Fonction pour créer un graphique d'analyse de campagne
@st.cache_data
def create_campaign_analysis(df):
    
    #Crée des visualisations pour analyser l'efficacité des campagnes de don.
    
    # Identifier les colonnes de date
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    # Sélectionner une colonne de date appropriée
    selected_date_col = None
    for col in date_columns:
        if df[col].notna().sum() > 100:  # Vérifier qu'il y a suffisamment de données
            selected_date_col = col
            break
    
    if selected_date_col:
        # Créer une copie du DataFrame avec la colonne de date
        df_temp = df.copy()
        
        # Convertir en datetime si ce n'est pas déjà fait
        df_temp[selected_date_col] = pd.to_datetime(df_temp[selected_date_col], errors='coerce')
        
        # Extraire l'année et le mois
        df_temp['year'] = df_temp[selected_date_col].dt.year
        df_temp['month'] = df_temp[selected_date_col].dt.month
        df_temp['year_month'] = df_temp[selected_date_col].dt.to_period('M')
        
        # Compter le nombre de donneurs par mois
        monthly_counts = df_temp.groupby('year_month').size().reset_index(name='count')
        monthly_counts['year_month_str'] = monthly_counts['year_month'].astype(str)
        
        # Créer un graphique de tendance temporelle
        fig1 = px.line(
            monthly_counts,
            x='year_month_str',
            y='count',
            markers=True,
            title=f"Évolution du nombre de donneurs au fil du temps (basé sur {selected_date_col})",
            labels={'count': 'Nombre de donneurs', 'year_month_str': 'Année-Mois'}
        )
        
        # Personnaliser le graphique
        fig1.update_layout(
            xaxis_title="Période",
            yaxis_title="Nombre de donneurs",
            font=dict(size=12),
            height=500
        )
        
        # Analyser les tendances par caractéristiques démographiques
        demographic_figs = []
        
        # Analyser par genre si disponible
        if 'Genre_' in df_temp.columns:
            gender_monthly = df_temp.groupby(['year_month', 'Genre_']).size().reset_index(name='count')
            gender_monthly['year_month_str'] = gender_monthly['year_month'].astype(str)
            
            fig_gender = px.line(
                gender_monthly,
                x='year_month_str',
                y='count',
                color='Genre_',
                markers=True,
                title="Évolution du nombre de donneurs par genre",
                labels={'count': 'Nombre de donneurs', 'year_month_str': 'Année-Mois', 'Genre_': 'Genre'}
            )
            
            fig_gender.update_layout(
                xaxis_title="Période",
                yaxis_title="Nombre de donneurs",
                legend_title="Genre",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_gender)
        
        # Analyser par niveau d'études si disponible
        if 'Niveau_d\'etude' in df_temp.columns:
            # Simplifier les catégories pour une meilleure lisibilité
            df_temp['Niveau_simplifié'] = df_temp['Niveau_d\'etude'].apply(
                lambda x: 'Universitaire' if 'Universitaire' in str(x) 
                else ('Secondaire' if 'Secondaire' in str(x)
                     else ('Primaire' if 'Primaire' in str(x)
                          else ('Aucun' if 'Aucun' in str(x) else 'Non précisé')))
            )
            
            edu_monthly = df_temp.groupby(['year_month', 'Niveau_simplifié']).size().reset_index(name='count')
            edu_monthly['year_month_str'] = edu_monthly['year_month'].astype(str)
            
            fig_edu = px.line(
                edu_monthly,
                x='year_month_str',
                y='count',
                color='Niveau_simplifié',
                markers=True,
                title="Évolution du nombre de donneurs par niveau d'études",
                labels={'count': 'Nombre de donneurs', 'year_month_str': 'Année-Mois', 'Niveau_simplifié': "Niveau d'études"}
            )
            
            fig_edu.update_layout(
                xaxis_title="Période",
                yaxis_title="Nombre de donneurs",
                legend_title="Niveau d'études",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_edu)
        
        return fig1, demographic_figs
    else:
        return None, []

# Fonction pour créer une analyse de fidélisation des donneurs
@st.cache_data
def create_donor_retention_analysis(df):
    
    #Crée des visualisations pour analyser la fidélisation des donneurs.
    
    # Vérifier si la colonne indiquant si le donneur a déjà donné est disponible
    if 'A-t-il_(elle)_déjà_donné_le_sang_' in df.columns:
        # Compter le nombre de donneurs qui ont déjà donné et ceux qui n'ont pas donné
        retention_counts = df['A-t-il_(elle)_déjà_donné_le_sang_'].value_counts().reset_index()
        retention_counts.columns = ['Statut', 'Nombre']
        
        # Créer un graphique circulaire
        fig1 = px.pie(
            retention_counts,
            values='Nombre',
            names='Statut',
            title="Proportion de donneurs fidélisés",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig1.update_layout(
            font=dict(size=12),
            height=400
        )
        
        # Analyser la fidélisation par caractéristiques démographiques
        demographic_figs = []
        
        # Analyser par genre si disponible
        if 'Genre_' in df.columns:
            gender_retention = pd.crosstab(df['Genre_'], df['A-t-il_(elle)_déjà_donné_le_sang_'])
            gender_retention_pct = gender_retention.div(gender_retention.sum(axis=1), axis=0) * 100
            
            # Convertir en format long pour Plotly
            gender_retention_long = gender_retention_pct.reset_index().melt(
                id_vars='Genre_',
                var_name='Statut',
                value_name='Pourcentage'
            )
            
            fig_gender = px.bar(
                gender_retention_long,
                x='Genre_',
                y='Pourcentage',
                color='Statut',
                barmode='group',
                title="Fidélisation des donneurs par genre",
                labels={'Pourcentage': 'Pourcentage (%)', 'Genre_': 'Genre', 'Statut': 'A déjà donné'}
            )
            
            fig_gender.update_layout(
                xaxis_title="Genre",
                yaxis_title="Pourcentage (%)",
                legend_title="A déjà donné",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_gender)
        
        # Analyser par niveau d'études si disponible
        if 'Niveau_d\'etude' in df.columns:
            # Simplifier les catégories pour une meilleure lisibilité
            df['Niveau_simplifié'] = df['Niveau_d\'etude'].apply(
                lambda x: 'Universitaire' if 'Universitaire' in str(x) 
                else ('Secondaire' if 'Secondaire' in str(x)
                     else ('Primaire' if 'Primaire' in str(x)
                          else ('Aucun' if 'Aucun' in str(x) else 'Non précisé')))
            )
            
            edu_retention = pd.crosstab(df['Niveau_simplifié'], df['A-t-il_(elle)_déjà_donné_le_sang_'])
            edu_retention_pct = edu_retention.div(edu_retention.sum(axis=1), axis=0) * 100
            
            # Convertir en format long pour Plotly
            edu_retention_long = edu_retention_pct.reset_index().melt(
                id_vars='Niveau_simplifié',
                var_name='Statut',
                value_name='Pourcentage'
            )
            
            fig_edu = px.bar(
                edu_retention_long,
                x='Niveau_simplifié',
                y='Pourcentage',
                color='Statut',
                barmode='group',
                title="Fidélisation des donneurs par niveau d'études",
                labels={'Pourcentage': 'Pourcentage (%)', 'Niveau_simplifié': "Niveau d'études", 'Statut': 'A déjà donné'}
            )
            
            fig_edu.update_layout(
                xaxis_title="Niveau d'études",
                yaxis_title="Pourcentage (%)",
                legend_title="A déjà donné",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_edu)
        
        # Analyser par âge si disponible
        if 'Age' in df.columns:
            # Créer des tranches d'âge
            df['Tranche_âge'] = pd.cut(
                df['Age'],
                bins=[0, 18, 25, 35, 45, 55, 100],
                labels=['<18', '18-25', '26-35', '36-45', '46-55', '>55']
            )
            
            age_retention = pd.crosstab(df['Tranche_âge'], df['A-t-il_(elle)_déjà_donné_le_sang_'])
            age_retention_pct = age_retention.div(age_retention.sum(axis=1), axis=0) * 100
            
            # Convertir en format long pour Plotly
            age_retention_long = age_retention_pct.reset_index().melt(
                id_vars='Tranche_âge',
                var_name='Statut',
                value_name='Pourcentage'
            )
            
            fig_age = px.bar(
                age_retention_long,
                x='Tranche_âge',
                y='Pourcentage',
                color='Statut',
                barmode='group',
                title="Fidélisation des donneurs par tranche d'âge",
                labels={'Pourcentage': 'Pourcentage (%)', 'Tranche_âge': "Tranche d'âge", 'Statut': 'A déjà donné'}
            )
            
            fig_age.update_layout(
                xaxis_title="Tranche d'âge",
                yaxis_title="Pourcentage (%)",
                legend_title="A déjà donné",
                font=dict(size=12),
                height=400
            )
            
            demographic_figs.append(fig_age)
        
        return fig1, demographic_figs
    else:
        return None, []

# Fonction pour créer une analyse de sentiment
@st.cache_data
def create_sentiment_analysis(df):
    #Crée des visualisations pour l'analyse de sentiment des commentaires des donneurs.
    # Vérifier si une colonne de commentaires est disponible
    comment_columns = [col for col in df.columns if any(term in col.lower() for term in 
                      ['préciser', 'raison', 'commentaire', 'feedback'])]
    
    if comment_columns:
        selected_col = comment_columns[0]
        
        # Filtrer les commentaires non vides
        comments_df = df[df[selected_col].notna() & (df[selected_col] != '')].copy()
        
        if len(comments_df) > 0:
            # Télécharger les ressources NLTK si nécessaire
            try:
                nltk.download('vader_lexicon', quiet=True)
                
                # Initialiser l'analyseur de sentiment
                sia = SentimentIntensityAnalyzer()
                
                # Fonction pour classifier le sentiment
                def classify_sentiment(text):
                    if pd.isna(text) or text == '':
                        return 'Neutre'
                    
                    # Utiliser TextBlob pour l'analyse de sentiment en français
                    blob = TextBlob(str(text))
                    polarity = blob.sentiment.polarity
                    
                    if polarity > 0.1:
                        return 'Positif'
                    elif polarity < -0.1:
                        return 'Négatif'
                    else:
                        return 'Neutre'
                
                # Appliquer l'analyse de sentiment
                comments_df['Sentiment'] = comments_df[selected_col].apply(classify_sentiment)
                
                # Compter les sentiments
                sentiment_counts = comments_df['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Nombre']
                
                # Créer un graphique circulaire
                fig1 = px.pie(
                    sentiment_counts,
                    values='Nombre',
                    names='Sentiment',
                    title="Analyse de sentiment des commentaires",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig1.update_layout(
                    font=dict(size=12),
                    height=400
                )
                
                # Créer un nuage de mots des commentaires les plus fréquents
                from wordcloud import WordCloud
                
                # Combiner tous les commentaires
                all_comments = ' '.join(comments_df[selected_col].astype(str).tolist())
                
                # Créer un nuage de mots
                wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_comments)
                
                # Convertir le nuage de mots en image
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                
                # Sauvegarder l'image
                wordcloud_path = r"C:\Users\Ultra Tech\Desktop\wordcloud.png"
                plt.savefig(wordcloud_path)
                plt.close()
                
                return fig1, wordcloud_path, comments_df
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de sentiment: {e}")
                return None, None, None
        else:
            return None, None, None
    else:
        return None, None, None




# Interface principale du tableau de bord
def main():
    #image_file=r"C:\Users\hp\Desktop\Projet dashboard\WhatsApp Image 2025-03-23 à 22.21.36_abdc063e.jpg"
    #set_background(image_file)
    
    #Fonction principale qui crée l'interface du tableau de bord Streamlit.
    
    # Titre et introduction
    st.title("📊 Tableau de Bord d'Analyse des Donneurs de Sang")
    st.markdown("""Ce tableau de bord interactif présente une analyse approfondie des données de donneurs de sang,
    permettant d'optimiser les campagnes de don et d'améliorer la gestion des donneurs.
    """)
    
    # Charger les données
    df_2019, df_volontaire,df,df_volontaires=load_data()
    
    # Barre latérale pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Sélectionnez une page",
        ["Aperçu des données", "Distribution géographique", "Santé et éligibilité", 
         "Profils des donneurs", "Analyse des campagnes", "Fidélisation des donneurs",
         "Analyse de sentiment", "Prédiction d'éligibilité"]
    )
    
    # Sélection du jeu de données
    st.sidebar.title("Jeu de données")
    dataset = st.sidebar.radio(
        "Sélectionnez un jeu de données",
        ["2019", "Volontaire"]
    )
    
    # Sélectionner le DataFrame en fonction du choix
    df = df_2019 if dataset == "2019" else df_volontaire
    
    # Afficher la page sélectionnée
    if page == "Aperçu des données":
        st.header("📋 Aperçu des données")
        
        # Afficher des statistiques générales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre total de donneurs", len(df))
        with col2:
            if 'Genre_' in df.columns:
                gender_counts = df['Genre_'].value_counts()
                st.metric("Hommes", gender_counts.get('Homme', 0))
        with col3:
            if 'Genre_' in df.columns:
                st.metric("Femmes", gender_counts.get('Femme', 0))
        
        # Afficher la distribution par âge si disponible
        if 'Age' in df.columns:
            st.subheader("Distribution par âge")
            fig = px.histogram(
                df,
                x='Age',
                nbins=20,
                title="Distribution des âges des donneurs",
                labels={'Age': 'Âge', 'count': 'Nombre de donneurs'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Afficher la distribution par niveau d'études si disponible
        if 'Niveau_d\'etude' in df.columns:
            st.subheader("Distribution par niveau d'études")
            edu_counts = df['Niveau_d\'etude'].value_counts().reset_index()
            edu_counts.columns = ['Niveau', 'Nombre']
            
            fig = px.bar(
                edu_counts,
                x='Niveau',
                y='Nombre',
                title="Distribution par niveau d'études",
                labels={'Niveau': "Niveau d'études", 'Nombre': 'Nombre de donneurs'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Afficher un aperçu du DataFrame
        st.subheader("Aperçu des données brutes")
        st.dataframe(paginate_dataframe(df))
        
        # Afficher des informations sur les colonnes
        st.subheader("Informations sur les colonnes")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    
    elif page == "Distribution géographique":
        st.header("🗺️ Distribution géographique des donneurs")
        
        # Identifier les colonnes géographiques
        geo_columns = [col for col in df.columns if any(term in col for term in 
                      ['Arrondissement', 'Quartier', 'Résidence'])]
        
        if geo_columns:
            # Sélectionner la colonne géographique
            geo_col = st.selectbox(
                "Sélectionnez une colonne géographique",
                geo_columns
            )
            
            # Sélectionner une colonne pour la coloration (optionnel)
            color_columns = ['ÉLIGIBILITÉ_AU_DON.'] if 'ÉLIGIBILITÉ_AU_DON.' in df.columns else []
            color_col = None
            if color_columns:
                color_option = st.selectbox(
                    "Colorer par (optionnel)",
                    ["Aucun"] + color_columns
                )
                if color_option != "Aucun":
                    color_col = color_option
            
            # Créer la carte
            st.subheader(f"Carte de distribution des donneurs par {geo_col}")
            m = create_geo_map(df, geo_col, color_col)
            folium_static(m)
            
            # Afficher un graphique à barres de la distribution
            st.subheader(f"Distribution des donneurs par {geo_col}")
            geo_counts = df[geo_col].value_counts().reset_index()
            geo_counts.columns = ['Zone', 'Nombre']
            
            # Limiter à 15 zones pour la lisibilité
            if len(geo_counts) > 15:
                geo_counts = geo_counts.head(15)
                st.info("Affichage des 15 zones les plus fréquentes uniquement.")
            
            fig = px.bar(
                geo_counts,
                x='Nombre',
                y='Zone',
                orientation='h',
                title=f"Distribution des donneurs par {geo_col}",
                labels={'Zone': geo_col, 'Nombre': 'Nombre de donneurs'}
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyser la relation entre la géographie et l'éligibilité si disponible
            if 'ÉLIGIBILITÉ_AU_DON.' in df.columns:
                st.subheader(f"Éligibilité au don par {geo_col}")
                
                # Sélectionner les 10 zones les plus fréquentes
                top_zones = geo_counts['Zone'].head(10).tolist()
                
                # Filtrer le DataFrame pour ces zones
                df_top_zones = df[df[geo_col].isin(top_zones)]
                
                # Créer une table de contingence
                contingency = pd.crosstab(df_top_zones[geo_col], df_top_zones['ÉLIGIBILITÉ_AU_DON.'])
                
                # Calculer les pourcentages par ligne
                contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                
                # Convertir en format long pour Plotly
                contingency_long = contingency_pct.reset_index().melt(
                    id_vars=geo_col,
                    var_name='Éligibilité',
                    value_name='Pourcentage'
                )
                
                fig = px.bar(
                    contingency_long,
                    x=geo_col,
                    y='Pourcentage',
                    color='Éligibilité',
                    barmode='stack',
                    title=f"Éligibilité au don par {geo_col}",
                    labels={'Pourcentage': 'Pourcentage (%)', geo_col: geo_col, 'Éligibilité': 'Éligibilité'}
                )
                
                fig.update_layout(
                    xaxis_title=geo_col,
                    yaxis_title="Pourcentage (%)",
                    legend_title="Éligibilité",
                    font=dict(size=12),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune colonne géographique identifiée dans les données.")
    
    elif page == "Santé et éligibilité":
        st.header("🩺 Conditions de santé et éligibilité au don")
        
        if 'ÉLIGIBILITÉ_AU_DON.' in df.columns:
            # Afficher des statistiques générales sur l'éligibilité
            st.subheader("Répartition de l'éligibilité au don")
            eligibility_counts = df['ÉLIGIBILITÉ_AU_DON.'].value_counts().reset_index()
            eligibility_counts.columns = ['Statut', 'Nombre']
            
            fig = px.pie(
                eligibility_counts,
                values='Nombre',
                names='Statut',
                title="Répartition de l'éligibilité au don",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Créer un graphique montrant l'impact des conditions de santé sur l'éligibilité
            st.subheader("Impact des conditions de santé sur l'éligibilité")
            health_fig = create_health_eligibility_chart(df)
            st.plotly_chart(health_fig, use_container_width=True)
            
            # Analyser l'impact des facteurs démographiques sur l'éligibilité
            st.subheader("Impact des facteurs démographiques sur l'éligibilité")
            
            # Sélectionner le facteur démographique
            demo_columns = [col for col in df.columns if any(term in col for term in 
                           ['Genre', 'Age', 'Niveau', 'Situation', 'Profession'])]
            
            if demo_columns:
                demo_col = st.selectbox(
                    "Sélectionnez un facteur démographique",
                    demo_columns
                )
                
                # Créer une table de contingence
                contingency = pd.crosstab(df[demo_col], df['ÉLIGIBILITÉ_AU_DON.'])
                
                # Calculer les pourcentages par ligne
                contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                
                # Convertir en format long pour Plotly
                contingency_long = contingency_pct.reset_index().melt(
                    id_vars=demo_col,
                    var_name='Éligibilité',
                    value_name='Pourcentage'
                )
                
                fig = px.bar(
                    contingency_long,
                    x=demo_col,
                    y='Pourcentage',
                    color='Éligibilité',
                    barmode='group',
                    title=f"Éligibilité au don par {demo_col}",
                    labels={'Pourcentage': 'Pourcentage (%)', demo_col: demo_col, 'Éligibilité': 'Éligibilité'}
                )
                
                fig.update_layout(
                    xaxis_title=demo_col,
                    yaxis_title="Pourcentage (%)",
                    legend_title="Éligibilité",
                    font=dict(size=12),
                    height=500,
                    xaxis={'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucun facteur démographique identifié dans les données.")
        else:
            st.warning("La colonne d'éligibilité au don n'est pas disponible dans ce jeu de données.")
    
    elif page == "Profils des donneurs":
        st.header("👥 Profils des donneurs")
        
        # Effectuer le clustering des donneurs
        cluster_fig, cluster_stats, df_with_clusters = create_donor_clustering(df)
        
        if cluster_fig is not None:
            # Afficher la visualisation des clusters
            st.subheader("Clustering des donneurs")
            st.plotly_chart(cluster_fig, use_container_width=True)
            
            # Afficher les caractéristiques des clusters
            st.subheader("Caractéristiques des clusters")
            st.dataframe(cluster_stats)
            
            # Créer des profils de donneurs idéaux
            st.subheader("Profils de donneurs idéaux")
            
            # Identifier le cluster avec le plus grand nombre de donneurs éligibles
            if 'ÉLIGIBILITÉ_AU_DON.' in df_with_clusters.columns and 'Cluster' in df_with_clusters.columns:
                # Compter le nombre de donneurs éligibles par cluster
                eligible_counts = df_with_clusters[df_with_clusters['ÉLIGIBILITÉ_AU_DON.'] == 'Eligible'].groupby('Cluster').size()
                
                if not eligible_counts.empty:
                    ideal_cluster = eligible_counts.idxmax()
                    
                    st.write(f"**Cluster idéal identifié: Cluster {ideal_cluster}**")
                    
                    # Extraire les caractéristiques du cluster idéal
                    ideal_profile = df_with_clusters[df_with_clusters['Cluster'] == ideal_cluster]
                    
                    # Afficher les caractéristiques démographiques du cluster idéal
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Caractéristiques démographiques:**")
                        
                        if 'Age' in ideal_profile.columns:
                            st.write(f"- Âge moyen: {ideal_profile['Age'].mean():.1f} ans")
                        
                        if 'Genre_' in ideal_profile.columns:
                            gender_pct = ideal_profile['Genre_'].value_counts(normalize=True) * 100
                            st.write(f"- Genre: {gender_pct.get('Homme', 0):.1f}% Hommes, {gender_pct.get('Femme', 0):.1f}% Femmes")
                        
                        if 'Niveau_d\'etude' in ideal_profile.columns:
                            top_edu = ideal_profile['Niveau_d\'etude'].value_counts(normalize=True).head(2)
                            st.write("- Niveau d'études principal:")
                            for edu, pct in top_edu.items():
                                st.write(f"  • {edu}: {pct*100:.1f}%")
                        
                        if 'Situation_Matrimoniale_(SM)' in ideal_profile.columns:
                            top_marital = ideal_profile['Situation_Matrimoniale_(SM)'].value_counts(normalize=True).head(2)
                            st.write("- Situation matrimoniale principale:")
                            for status, pct in top_marital.items():
                                st.write(f"  • {status}: {pct*100:.1f}%")
                    
                    with col2:
                        st.write("**Caractéristiques géographiques:**")
                        
                        geo_columns = [col for col in ideal_profile.columns if any(term in col for term in 
                                      ['Arrondissement', 'Quartier', 'Résidence'])]
                        
                        for geo_col in geo_columns:
                            top_geo = ideal_profile[geo_col].value_counts(normalize=True).head(3)
                            st.write(f"- {geo_col} principal:")
                            for zone, pct in top_geo.items():
                                st.write(f"  • {zone}: {pct*100:.1f}%")
                    
                    # Créer un radar chart pour visualiser le profil idéal
                    if 'Age' in ideal_profile.columns and 'Taille_' in ideal_profile.columns and 'Poids' in ideal_profile.columns:
                        # Calculer les moyennes normalisées
                        avg_age = ideal_profile['Age'].mean() / df['Age'].max()
                        avg_height = ideal_profile['Taille_'].mean() / df['Taille_'].max()
                        avg_weight = ideal_profile['Poids'].mean() / df['Poids'].max()
                        
                        # Créer un radar chart
                        categories = ['Âge', 'Taille', 'Poids']
                        values = [avg_age, avg_height, avg_weight]
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Profil idéal'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Caractéristiques physiques du profil idéal (normalisées)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Impossible de déterminer le profil idéal car la colonne d'éligibilité n'est pas disponible.")
        else:
            st.warning("Impossible d'effectuer le clustering car il n'y a pas assez de variables numériques dans les données.")
    
    elif page == "Analyse des campagnes":
        st.header("📈 Analyse des campagnes de don")
        
        # Créer des visualisations pour l'analyse des campagnes
        campaign_fig, demographic_figs = create_campaign_analysis(df)
        
        if campaign_fig is not None:
            # Afficher la tendance temporelle générale
            st.subheader("Évolution du nombre de donneurs au fil du temps")
            st.plotly_chart(campaign_fig, use_container_width=True)
            
            # Afficher les tendances par caractéristiques démographiques
            if demographic_figs:
                st.subheader("Tendances par caractéristiques démographiques")
                
                for fig in demographic_figs:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Ajouter des recommandations pour l'optimisation des campagnes
            st.subheader("Recommandations pour l'optimisation des campagnes")
            
            st.write("""Sur la base de l analyse des données, voici quelques recommandations pour optimiser les futures campagnes de don de sang:
            1. **Ciblage démographique**: Concentrez les efforts sur les segments de population les plus susceptibles de donner, comme identifié dans l'analyse des profils.
            
            2. **Planification temporelle**: Organisez les campagnes pendant les périodes où le taux de participation est historiquement élevé.
            
            3. **Localisation géographique**: Privilégiez les zones géographiques avec un taux d'éligibilité élevé et une forte concentration de donneurs potentiels.
            
            4. **Sensibilisation ciblée**: Développez des messages spécifiques pour les groupes sous-représentés afin d'augmenter leur participation.
            
            5. **Fidélisation**: Mettez en place des stratégies pour encourager les donneurs à revenir régulièrement.
            """)
        else:
            st.warning("Impossible d'analyser les tendances temporelles car aucune colonne de date appropriée n'a été identifiée.")
    
    elif page == "Fidélisation des donneurs":
        st.header("🔄 Fidélisation des donneurs")
        
        # Créer des visualisations pour l'analyse de fidélisation
        retention_fig, demographic_figs = create_donor_retention_analysis(df)
        
        if retention_fig is not None:
            # Afficher la proportion de donneurs fidélisés
            st.subheader("Proportion de donneurs fidélisés")
            st.plotly_chart(retention_fig, use_container_width=True)
            
            # Afficher la fidélisation par caractéristiques démographiques
            if demographic_figs:
                st.subheader("Fidélisation par caractéristiques démographiques")
                
                for fig in demographic_figs:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Ajouter des stratégies pour améliorer la fidélisation
            st.subheader("Stratégies pour améliorer la fidélisation des donneurs")
            
            st.write("""Voici quelques stratégies pour améliorer la fidélisation des donneurs de sang:
            
            1. **Programme de reconnaissance**: Mettre en place un système de reconnaissance pour les donneurs réguliers (badges, certificats, etc.).
            
            2. **Communication personnalisée**: Envoyer des rappels personnalisés aux donneurs en fonction de leur historique de don.
            
            3. **Expérience positive**: Améliorer l expérience du donneur pendant le processus de don pour encourager le retour.
            
            4. **Éducation continue**: Informer les donneurs sur l impact de leur don et l importance de donner régulièrement.
            
            5. **Événements communautaires**: Organiser des événements spéciaux pour les donneurs réguliers afin de renforcer leur engagement.
            """)
        else:
            st.warning("Impossible d'analyser la fidélisation car les informations nécessaires ne sont pas disponibles dans les données.")
    
    elif page == "Analyse de sentiment":
        st.header("💬 Analyse de sentiment des retours")
        #paths=os.path.join(os.getcwd(),"Challenge dataset traité.xlsx")
        paths="Challenge_dataset_traité.csv"
        df=pd.read_csv(paths)
        df_volontaires=pd.read_csv(paths)
        sia = SentimentIntensityAnalyzer()
        # Créer des visualisations pour l'analyse de sentiment
        sentiment_fig, wordcloud_path, comments_df = create_sentiment_analysis(df)
        if "Autres_Raisons_Precises" in df_volontaires.columns:
            df_volontaires["Sentiment"] = df_volontaires["Autres_Raisons_Precises"].dropna().apply(lambda x: sia.polarity_scores(str(x))["compound"])
            sentiment_counts = df_volontaires["Sentiment"].apply(lambda x: "Positif" if x > 0 else "Négatif" if x < 0 else "Neutre").value_counts()
            fig = px.pie(sentiment_counts, names=sentiment_counts.index, title="Répartition des sentiments")
            st.plotly_chart(fig)
            text = " ".join(str(f) for f in df_volontaires["Autres_Raisons_Precises"].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            st.image(wordcloud.to_array(), caption="Nuage de Mots des Feedbacks", use_column_width=True)
        
        
        df["Sentiment"] = df["Autres_Raisons_Precises"].fillna("").apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df["Sentiment_Catégorie"] = df["Sentiment"].apply(lambda x: "Positif" if x > 0 else "Négatif" if x < 0 else "Neutre")

        fig_sentiment = px.histogram(df, x="Sentiment_Catégorie", title="Répartition des sentiments des feedbacks")
        st.plotly_chart(fig_sentiment)

        st.write("*Note :* Les valeurs positives indiquent des feedbacks positifs, les négatives des plaintes.")

            
    
    elif page == "Prédiction d'éligibilité":
        st.header("🔮 Prédiction d'éligibilité au don")
        
        st.title("🔍 Prédiction d'Éligibilité au Don de Sang")
        # ==============================
        # 🎯 CHARGEMENT DU MODÈLE & ENCODEURS
        # ==============================
        @st.cache_resource
        def load_model():
            pathse="eligibility_model.pkl"
            data = joblib.load(pathse)
            return data["model"], data["X_test"], data["y_test"], data["target_encoder"], data["lpreprocessor"],data["resultat"]

        model, X_test, y_test, target_encoder, preprocessor,resultat = load_model()
        

        columns_binary = [
                "Drepanocytose", "Opere", "Transfusion_Antecedent", "Diabete",
                "Hypertension", "Porteur_VIH_HBS_HCV", "Asthme",
                "Probleme_Cardiaque", "Tatouage", "Scarification", "Deja_Donneur"
            ]



        # ==============================
        # 🗂 ORGANISATION EN ONGLETS
        # ==============================
        tab1, tab2,tab3= st.tabs([ "🔄 Prédiction Individuelle","📥 Télécharger/Charger un Fichier","Performance du modèle"])


        with tab1:
            col1,col2,col3=st.columns(3)
            st.subheader("🔄 Faire une prédiction individuelle")

            st.write("""Ce modèle prédit si un donneur est éligible ou non en fonction de ses caractéristiques médicales et personnelles.
            Remplissez les informations ci-dessous pour obtenir une prédiction.
        """)
            df=pd.read_csv("Challenge_dataset_traité.csv")
            # ==============================
            # 📌 FORMULAIRE DE SAISIE
            # ==============================
            with col1:
                age = st.number_input("Âge", min_value=18, max_value=100, value=30, step=1)

                profession = st.selectbox("Profession",list(df["Profession"].dropna().unique()) )
                religion = st.selectbox("Religion", list(df["Religion"].dropna().unique()))
                Niveau_Etude =st.selectbox("Religion", list(df["Niveau_Etude"].dropna().unique()))
                Statut_Matrimonial= st.selectbox("Religion", list(df["Statut_Matrimonial"].dropna().unique()))
                # État de santé (Binaire : Oui/Non)
            columns_binary = [
                "Drepanocytose", "Opere", "Transfusion_Antecedent", "Diabete",
                "Hypertension", "Porteur_VIH_HBS_HCV", "Asthme",
                "Probleme_Cardiaque", "Tatouage", "Scarification", "Deja_Donneur"
            ]
            columns_binarys=[
                "Drepanocytose", "Opere", "Transfusion_Antecedent", "Diabete",
                "Hypertension", "Porteur_VIH_HBS_HCV"]
            columns_binaryss=["Asthme",
                "Probleme_Cardiaque", "Tatouage", "Scarification", "Deja_Donneur"]
            
            with col2:
                binary_inputs = {}
                for col in columns_binarys:
                    binary_inputs[col] = st.radio(col, ["Non", "Oui"])
            with col3:
                for col in columns_binaryss:
                    binary_inputs[col] = st.radio(col, ["Non", "Oui"])

            # ==============================
            # 🔄 PRÉPARATION DES DONNÉES
            # ==============================
            # Convertir les entrées utilisateur en dataframe
            input_data = pd.DataFrame([[age, profession] + [binary_inputs[col] for col in columns_binary]+[religion,Niveau_Etude,Statut_Matrimonial]],
                                    columns=["Age", "Profession"] + columns_binary+["Religion","Niveau_Etude","Statut_Matrimonial"])

            # Encoder Profession et Religion
            input_data =preprocessor.transform(input_data)
            # ==============================
            # 🚀 PRÉDICTION
            # ==============================
            if st.button("Prédire l'éligibilité"):
                prediction = model.predict(input_data)[0]
                result = target_encoder.inverse_transform([prediction])[0]
                 # Afficher des recommandations en fonction de la prédiction
                st.subheader("Recommandations")
                


                # Affichage du résultat
                if prediction == 1:
                    st.success("✅ Le donneur est *ÉLIGIBLE* au don de sang !")
                    st.balloons()
                else:
                    st.error("❌ Le donneur *N'EST PAS ÉLIGIBLE* au don de sang.")

                # Affichage des valeurs d'entrée encodées
                st.subheader("🔎 Données encodées utilisées pour la prédiction :")
                #st.dataframe(input_data)
        
        
        # ==============================
        # 📥 ONGLET : TÉLÉCHARGEMENT / UPLOAD
        # ==============================
        with tab2:

            # ==============================
            # 📥 TÉLÉCHARGEMENT DU FICHIER MODÈLE
            # ==============================
            st.subheader("📥 Télécharger le modèle de fichier Excel")
            sample_data = pd.DataFrame({
                "Age": [30, 45],
                "Profession": ["Médecin", "Enseignant"],
                "Drepanocytose": ["Non", "Oui"],
                "Opere": ["Non", "Non"],
                "Transfusion_Antecedent": ["Oui", "Non"],
                "Diabete": ["Non", "Oui"],
                "Hypertension": ["Oui", "Non"],
                "Porteur_VIH_HBS_HCV": ["Non", "Non"],
                "Asthme": ["Non", "Oui"],
                "Probleme_Cardiaque": ["Non", "Non"],
                "Tatouage": ["Oui", "Non"],
                "Scarification": ["Non", "Oui"],
                "Deja_Donneur": ["Oui", "Non"],
                "Religion": ["Musulman", "Chrétien"],
                "Niveau_Etude":["Primaire","secondaire"],
                "Statut_Matrimonial":["Marier","Celibataire"]
            })

            # Génération du fichier Excel à télécharger
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                sample_data.to_excel(writer, index=False, sheet_name="Exemple")
                writer.close()

            st.download_button(
                label="📥 Télécharger le fichier modèle",
                data=output.getvalue(),
                file_name="Modele_Donneurs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ==============================
            # 📤 UPLOADER UN FICHIER EXCEL AVEC LES DONNEURS
            # ==============================
            st.subheader("📤 Uploader un fichier Excel contenant les informations des donneurs")
            uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx"])

            if uploaded_file:
                # Lecture du fichier
                df_uploaded = pd.read_excel(uploaded_file)

                # Vérification des colonnes attendues
                expected_columns = sample_data.columns.tolist()
                if not all(col in df_uploaded.columns for col in expected_columns):
                    st.error("⚠ Le fichier ne contient pas les bonnes colonnes ! Vérifiez le format et réessayez.")
                else:
                    st.success("✅ Fichier chargé avec succès !")

                    # ==============================
                    # 🔄 PRÉTRAITEMENT DES DONNÉES POUR TOUTES LES LIGNES
                    # ==============================
                    # Supprimer les lignes avec des valeurs non reconnues
                    df_uploaded.dropna(inplace=True)
                    colonne=["Age", "Profession", "Drepanocytose","Opere",
                "Transfusion_Antecedent","Diabete", "Hypertension", "Porteur_VIH_HBS_HCV", 
                "Asthme", "Probleme_Cardiaque","Tatouage","Scarification", "Deja_Donneur", "Religion","Niveau_Etude","Statut_Matrimonial"]
                    input_data =preprocessor.transform(df_uploaded)
                    df_uploadeds=input_data.copy()
                    

                    # Encoder les valeurs binaires (Oui = 1, Non = 0)
                    if df_uploadeds.shape[0] == 0:
                        st.error("Aucune donnée valide après prétraitement ! vérifiez les données d'entrée")

                    # ==============================
                    # 🚀 PRÉDICTION SUR TOUTES LES LIGNES
                    # ==============================
                    df_uploaded["Prédiction"] = model.predict(df_uploadeds)

                    # Décodage des résultats
                    df_uploaded["Prédiction"] = df_uploaded["Prédiction"].map(lambda x: target_encoder.inverse_transform([x])[0])

                    # Affichage du tableau avec les prédictions
                    st.subheader("📊 Résultats des prédictions")
                    st.dataframe(df_uploaded)

                    # ==============================
                    # 📤 TÉLÉCHARGER LE FICHIER AVEC PREDICTIONS
                    # ==============================
                    output_predictions = io.BytesIO()
                    with pd.ExcelWriter(output_predictions, engine='xlsxwriter') as writer:
                        df_uploaded.to_excel(writer, index=False, sheet_name="Prédictions")
                        writer.close()

                    st.download_button(
                        label="📥 Télécharger les résultats",
                        data=output_predictions.getvalue(),
                        file_name="Resultats_Predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    # ==============================
    # 📈 ONGLET : PERFORMANCE DU MODÈLE
    # ==============================
        with tab3:
            # ==============================
            # 🎯 CHARGEMENT DU MODÈLE & DONNÉES TEST
            # ==============================
            @st.cache_resource
            def load_model():
               pathse="eligibility_model.pkl"
               data = joblib.load(pathse)
               return data["model"], data["X_test"], data["y_test"], data["target_encoder"], data["lpreprocessor"],data["resultat"]

            model, X_test, y_test, target_encoder, preprocessor,resultat = load_model()
                # ==============================
            # 📊 PERFORMANCE DU MODÈLE SUR DONNÉES TEST
            # ==============================
            st.subheader("📈 Performance du Modèle sur Données de Test")

            # 🔮 Prédictions sur X_test
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # 📄 Rapport de Classification
            st.subheader("📄 Rapport de Classification")
            A=target_encoder.inverse_transform([0,1,2])
            report = classification_report(y_test, y_pred, target_names=A, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)
            st.dataframe(resultat)
            

            # 🔄 Binarisation des étiquettes pour "One vs Rest" si multi-classe
            n_classes = len(np.unique(y_test))  # Nombre de classes
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))  # Transforme y_test en binaire
            y_pred_proba = model.predict_proba(X_test)  # Probabilités prédites pour chaque classe

            # 📈 Affichage de la courbe ROC pour chaque classe
            st.subheader("📈 Courbe ROC (One vs Rest)")

            fig, ax = plt.subplots(figsize=(7, 5))
            colors = ['blue', 'red', 'green', 'purple', 'orange']  # Couleurs pour chaque classe

            for i in range(n_classes):
                if n_classes > 2:  # Cas multi-classe
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"Classe {A[i]} (AUC = {roc_auc:.2f})")
                else:  # Cas binaire classique
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")

            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.set_xlabel("Taux de Faux Positifs (FPR)")
            ax.set_ylabel("Taux de Vrais Positifs (TPR)")
            ax.set_title("Courbe ROC par Classe")
            ax.legend(loc="lower right")
            st.pyplot(fig)

            
            # 📊 Matrice de Confusion
            st.subheader("📊 Matrice de Confusion")
            B=target_encoder.inverse_transform(y_test)
            C=target_encoder.inverse_transform(y_pred)
            conf_matrix = confusion_matrix(B, C)
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Matrice de Confusion")
            st.pyplot(fig)
            # Afficher les probabilités
            st.write("**Probabilités:**")
            
            # Créer un DataFrame pour les probabilités
            proba_df = pd.DataFrame({
                'Statut': model.classes_,
                'Probabilité': y_pred_proba[0]
            })
            
            # Créer un graphique à barres pour les probabilités
            fig = px.bar(
                proba_df,
                x='Statut',
                y='Probabilité',
                title="Probabilités pour chaque statut d'éligibilité",
                labels={'Probabilité': 'Probabilité', 'Statut': "Statut d'éligibilité"}
            )
            
            fig.update_layout(
                xaxis_title="Statut d'éligibilité",
                yaxis_title="Probabilité",
                font=dict(size=12),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

            #if hasattr(model, 'feature_importances_'):
            # Obtenir les noms des caractéristiques après one-hot encoding
            feature_names = []
            for name, transformer, features in preprocessor.transformers_:
                if name == 'cat':
                    # Pour les caractéristiques catégorielles, obtenir les noms après one-hot encoding
                    for i, feature in enumerate(features):
                        categories = transformer.named_steps['onehot'].categories_[i]
                        for category in categories:
                            feature_names.append(f"{feature}_{category}")
                else:
                    # Pour les caractéristiques numériques, conserver les noms d'origine
                    feature_names.extend(features)
            
            # Obtenir les importances des caractéristiques
            importances = model.feature_importances_
            
            # Créer un DataFrame pour les importances
            if len(feature_names) == len(importances):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                
                # Trier par importance décroissante
                importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
                
                # Créer un graphique des importances
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Importance des caractéristiques pour la prédiction d'éligibilité",
                    labels={'Importance': 'Importance relative', 'Feature': 'Caractéristique'}
                )
                
                fig_importance.update_layout(
                    xaxis_title="Importance relative",
                    yaxis_title="Caractéristique",
                    font=dict(size=12),
                    height=600
                )
            st.plotly_chart(fig_importance)
           
                    
                    
    
    # Pied de page
    st.markdown("---")
    st.markdown(""" <div style="text-align: center;">
        <p>Tableau de bord développé pour le concours de data visualisation sur les donneurs de sang</p>
        <p>© 2025 - Tous droits réservés</p>
    </div>
    """, unsafe_allow_html=True)

# Point d'entrée principal
if __name__ == "__main__":
    main()
