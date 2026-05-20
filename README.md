# Système Intelligent Multi-Modèles pour la Rétention Client

Projet Data Science — EFREI M1 Data Engineering  
Prédiction du risque de résiliation client (Customer Churn) avec comparaison multi-modèles, API REST et dashboard décisionnel interactif.

---

## Prérequis — Télécharger le dataset

Le fichier de données n'est pas inclus dans le dépôt. Avant de lancer le projet :

1. Télécharger le dataset sur Kaggle : [customer-churn-prediction-business-dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
2. Télécharger le fichier `customer_churn_business_dataset.csv`
3. Le placer dans le dossier `data/` du projet :

```
Projet_Datascience/
└── data/
    └── customer_churn_business_dataset.csv   ← ici
```

---

## Lancement rapide avec Docker

Prérequis : avoir [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et lancé.

```bash
docker-compose up --build
```

C'est tout. Une fois les conteneurs démarrés :

| Service | URL |
|---|---|
| Dashboard Streamlit | http://localhost:8501 |
| API FastAPI (docs Swagger) | http://localhost:8000/docs |
| Health check API | http://localhost:8000/health |

Pour arrêter :

```bash
docker-compose down
```

> Le dashboard attend automatiquement que l'API soit prête avant de démarrer (healthcheck intégré).  
> Le dashboard appelle l'API pour toutes les prédictions. Si l'API est indisponible, il bascule sur le modèle local.

---

## Structure du projet

```
Projet_Datascience/
│
├── api/
│   └── main.py               # API REST FastAPI (endpoints /predict, /health, /model-info)
│
├── dashboard/
│   └── app.py                # Dashboard Streamlit (4 onglets : Vue globale, Risques, Simulateur, Modèles)
│
├── data/
│   └── customer_churn_business_dataset.csv   # Dataset 10 000 clients
│
├── models/
│   ├── final_model.joblib        # Pipeline XGBoost entraîné (StandardScaler + XGBClassifier)
│   ├── expected_columns.joblib   # Colonnes attendues pour l'inférence
│   └── comparison_results.json   # Métriques des 4 modèles (généré par le notebook 02)
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb     # Analyse exploratoire, corrélations, préparation
│   └── 02_Modeling_and_Evaluation.ipynb   # Entraînement, comparaison, SHAP
│
├── src/
│   ├── data_processing.py   # Chargement et encodage des données
│   └── train.py             # Script d'entraînement du modèle de production
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Architecture

```
Streamlit Dashboard  →  POST /predict  →  FastAPI API  →  final_model.joblib
     (port 8501)                            (port 8000)       (XGBoost pipeline)
```

Dans Docker, le dashboard communique avec l'API via le réseau interne (`http://api:8000`).  
En local sans Docker, il appelle `http://localhost:8000`.

---

## Lancement en local (sans Docker)

Si Docker n'est pas disponible, voici les étapes manuelles.

### 1. Environnement Python

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Entraîner le modèle

Le modèle est déjà présent dans `models/`. Si vous souhaitez le ré-entraîner :

```bash
python src/train.py
```

### 3. Lancer l'API (terminal 1)

```bash
uvicorn api.main:app --reload
```

### 4. Lancer le dashboard (terminal 2)

```bash
streamlit run dashboard/app.py
```

---

## Modèles comparés

| Modèle | Type | Rôle |
|---|---|---|
| Régression Logistique | ML classique | Baseline interprétable |
| Random Forest | ML classique | Capture les non-linéarités |
| XGBoost | Gradient Boosting | Modèle de production retenu |
| MLP | Deep Learning (sklearn) | Comparaison avec réseau de neurones |

Le modèle final retenu est **XGBoost** — meilleur compromis performance / stabilité / interprétabilité.  
Les résultats complets de comparaison sont visibles dans l'onglet **Performance des Modèles** du dashboard.

---

## Interprétabilité

Les notebooks incluent une analyse **SHAP** (SHapley Additive exPlanations) sur XGBoost :
- Quelles variables influencent le plus les prédictions
- Dans quel sens (augmente ou réduit le risque de churn)
- Exploitable directement par une équipe CRM ou marketing

---

## Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Vérifie que l'API et le modèle sont chargés |
| GET | `/model-info` | Type de modèle, nombre de features, steps du pipeline |
| POST | `/predict` | Reçoit les features d'un client, retourne `prediction`, `churn_probability`, `risk_level` |

Exemple de requête `POST /predict` :

```json
{
  "features": {
    "age": 35,
    "tenure_months": 6,
    "monthly_fee": 80,
    "total_revenue": 480,
    "csat_score": 2.5,
    "contract_type": "Monthly",
    "payment_failures": 3,
    "support_tickets": 4,
    "usage_growth_rate": -0.2
  }
}
```

Réponse :

```json
{
  "prediction": 1,
  "churn_probability": 0.7823,
  "risk_level": "eleve"
}
```

La documentation interactive complète est disponible sur http://localhost:8000/docs (Swagger UI).
