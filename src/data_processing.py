import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path):
    """
    Charge et prépare les données pour l'entraînement.
    """
    df = pd.read_csv(data_path)
    
    # Supression des identifiants
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
        
    # Variables catégorielles (One-Hot)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Séparation cible et features
    if 'churn' in df_encoded.columns:
        X = df_encoded.drop(columns=['churn'])
        y = df_encoded['churn']
    else:
        # En cas d'inférence, il n'y a pas de cible 'churn'
        X = df_encoded
        y = None
        
    return X, y

def get_feature_names(data_path):
    """
    Retourne les noms des colonnes attendues après encodage.
    """
    X, _ = load_and_preprocess_data(data_path)
    return X.columns.tolist()
