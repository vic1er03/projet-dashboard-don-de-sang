import pandas as pd
import joblib
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


# Charger les données nettoyées
data = pd.read_excel(r"C:\Users\hp\Desktop\Projet dashboard\Challenge dataset traité.xlsx")

# Sélectionner les nouvelles features
features = ["Age", "Profession", "Drepanocytose","Opere",
            "Transfusion_Antecedent","Diabete", "Hypertension", "Porteur_VIH_HBS_HCV", 
            "Asthme", "Probleme_Cardiaque","Tatouage","Scarification", "Deja_Donneur", "Religion","Niveau_Etude","Statut_Matrimonial"]
target = "Eligibilite_Don"
# Encoder la cible
target_encoder = LabelEncoder()
data[target] = target_encoder.fit_transform(data[target])  # 1 = Eligible, 0 = Non-Eligible
# Séparer les données
Xs = data[features].dropna()
y = data[target][Xs.index]
 # Identifier les types de colonnes
numeric_features = Xs.select_dtypes(include=['int64', 'float64']).columns
categorical_features = Xs.select_dtypes(include=['object', 'bool']).columns

# Créer des transformateurs pour chaque type de colonne
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combiner les transformateurs
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
X=preprocessor.fit_transform(Xs)

"""# Encoder les variables catégoriques
label_encoders = {}
for col in ["Profession", "Religion","Niveau_Etude","Statut_Matrimonial"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le  # Sauvegarde de l'encodeur pour l'API

# Encodage des états de santé (binaire 0/1)
for col in ["Drepanocytose","Opere","Transfusion_Antecedent","Diabete", "Hypertension", "Porteur_VIH_HBS_HCV", "Asthme", "Probleme_Cardiaque","Tatouage","Scarification", "Deja_Donneur"]:
    data[col] = data[col].map({"Oui": 1, "Non": 0})

# Encoder la cible
target_encoder = LabelEncoder()
data[target] = target_encoder.fit_transform(data[target])  # 1 = Eligible, 0 = Non-Eligible"""



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition des modèles et des hyperparamètres
models = {
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
    },
    "SVM": {
        "model": SVC(),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7]}
    },
    "LogisticRegression": {
        "model": LogisticRegression(),
        "params": {"C": [0.1, 1, 10]}
    }
}
# Stocker les résultats
best_model = None
best_score = 0
results = []

# Tester chaque modèle avec GridSearchCV
for name, model_info in models.items():
    clf = GridSearchCV(model_info["model"], model_info["params"], cv=5, scoring="f1")
    clf.fit(X_train, y_train)
    
    # Meilleur modèle et score
    best_model_params = clf.best_params_
    best_model_instance = clf.best_estimator_
    y_pred = best_model_instance.predict(X_test)
    score = f1_score(y_test, y_pred,average="weighted")

    # Stocker les résultats
    results.append({"Model": name, "Best Params": best_model_params, "F1 Score": score})

    # Vérifier si c'est le meilleur modèle
    if score > best_score:
        best_score = score
        best_model = (name, best_model_instance, best_model_params)
# Afficher les résultats
df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

# Entraîner un modèle Random Forest
model = best_model[1]


# Évaluer la performance
y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))

# Sauvegarder le modèle et les encodeurs
joblib.dump(target_encoder, r"C:\Users\hp\Desktop\Projet dashboard\target_encoder.pkl")
joblib.dump(label_encoders, r"C:\Users\hp\Desktop\Projet dashboard\label_encoders.pkl")

# Sauvegarder le modèle et les encodeurs
joblib.dump({
    "model": model,
    "X_test": X_test,
    "y_test": y_test,
    "target_encoder": target_encoder,
    "lpreprocessor": preprocessor,
    "resultat":df_results
}, r"C:\Users\hp\Desktop\Projet dashboard\eligibility_model.pkl")

print("✅ Modèle entraîné avec les données de test sauvegardées !")

print("✅ Modèle entraîné et sauvegardé avec les nouvelles variables !")