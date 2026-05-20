import os
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_processing import load_and_preprocess_data

def train_and_save_model(data_path, model_output_path):
    print("Chargement et préparation des données...")
    X, y = load_and_preprocess_data(data_path)
    
    print("Création du pipeline d'entraînement...")
    # Calcul du ratio pour compenser le desequilibre des classes (~10% de churn)
    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    scale_pos_weight = round(n0 / n1, 2)
    print(f"Desequilibre detecte : {n1} churners / {n0} non-churners -> scale_pos_weight={scale_pos_weight}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ))
    ])
    
    print("Entraînement du modèle XGBoost en cours...")
    pipeline.fit(X, y)
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    print(f"Sauvegarde du modèle dans {model_output_path}...")
    joblib.dump(pipeline, model_output_path)
    
    # On sauvegarde aussi les colonnes attendues pour l'inférence
    columns_path = os.path.join(os.path.dirname(model_output_path), "expected_columns.joblib")
    joblib.dump(X.columns.tolist(), columns_path)
    
    print("Entraînement et sauvegarde terminés avec succès.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn_business_dataset.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.joblib")
    train_and_save_model(DATA_PATH, MODEL_PATH)
