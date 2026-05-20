import os
import json
import joblib
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Retention Client - Dashboard", layout="wide")

sns.set_theme(style="whitegrid")

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "expected_columns.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn_business_dataset.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "comparison_results.json")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# CSS
st.markdown("""
<style>
    .kpi-box {
        background-color: #1e3a5f;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    .kpi-box-danger {
        background-color: #8b1a1a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    .kpi-title { font-size: 1rem; font-weight: 600; margin-bottom: 8px; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; }
    .section-header {
        background-color: #f0f2f6;
        border-left: 4px solid #1e3a5f;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-weight: 600;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    columns = joblib.load(COLUMNS_PATH) if os.path.exists(COLUMNS_PATH) else None
    data = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None
    results = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    return model, columns, data, results


model, expected_columns, data, comparison_results = load_assets()

if model is None or data is None:
    st.error("Modele ou donnees introuvables. Executez d'abord : python src/train.py")
    st.stop()


def predict_via_api(features: dict):
    """Appelle l'API pour obtenir la prediction. Retourne None si l'API n'est pas disponible."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=3
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def predict_local(features: dict):
    """Prediction directe via le modele (fallback si API indisponible)."""
    df_input = pd.DataFrame([features])
    df_encoded = pd.get_dummies(df_input)
    aligned = pd.DataFrame(0, index=[0], columns=expected_columns)
    for col in expected_columns:
        if col in df_encoded.columns:
            aligned[col] = df_encoded[col].values
    proba = model.predict_proba(aligned)[0][1]
    pred = int(model.predict(aligned)[0])
    return {
        "prediction": pred,
        "churn_probability": round(float(proba), 4),
        "risk_level": "eleve" if proba > 0.5 else "faible"
    }


def get_prediction(features: dict):
    """Essaie l'API, repli sur le modele local si indisponible."""
    result = predict_via_api(features)
    source = "API"
    if result is None:
        result = predict_local(features)
        source = "modele local (API non disponible)"
    return result, source


# ---- TITRE ----
st.title("Systeme Intelligent de Retention Client")
st.markdown("Tableau de bord decisionnel pour anticiper le risque de resiliation et estimer l'impact financier.")

# ---- VERIF API ----
api_status = predict_via_api({"age": 0})
if api_status is not None or True:
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("API connectee et operationnelle (http://localhost:8000)")
        else:
            st.warning("API detectee mais retourne une erreur.")
    except Exception:
        st.info("API non disponible — les predictions utilisent le modele local directement.")

st.divider()

# ---- ONGLETS ----
tab1, tab2, tab3, tab4 = st.tabs([
    "Vue Globale",
    "Analyse des Risques",
    "Simulateur Client",
    "Performance des Modeles"
])


# ==============================================================
# TAB 1 — VUE GLOBALE
# ==============================================================
with tab1:
    st.markdown('<div class="section-header">Indicateurs Cles (KPI)</div>', unsafe_allow_html=True)

    total_customers = data.shape[0]
    churn_rate = data["churn"].mean() if "churn" in data.columns else 0.0
    total_revenue = data["total_revenue"].sum() if "total_revenue" in data.columns else 0.0
    revenue_at_risk = total_revenue * churn_rate
    n_churners = int(data["churn"].sum()) if "churn" in data.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Clients Totaux</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Taux de Resiliation</div>
            <div class="kpi-value">{churn_rate*100:.1f} %</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-box-danger">
            <div class="kpi-title">Clients a Risque</div>
            <div class="kpi-value">{n_churners:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-box-danger">
            <div class="kpi-title">Revenu a Risque Estime</div>
            <div class="kpi-value">$ {revenue_at_risk:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Distribution des Clients</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Churn distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_counts = data["churn"].value_counts()
        colors = ["#1e3a5f", "#c0392b"]
        ax.bar(["Fideles", "Resilies"], churn_counts.values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title("Distribution Churn / Non-Churn", fontweight="bold")
        ax.set_ylabel("Nombre de clients")
        for i, v in enumerate(churn_counts.values):
            ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col_b:
        # Churn par type de contrat
        if "contract_type" in data.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ct = data.groupby(["contract_type", "churn"]).size().unstack(fill_value=0)
            ct.plot(kind="bar", ax=ax, color=["#1e3a5f", "#c0392b"], edgecolor="white")
            ax.set_title("Churn par Type de Contrat", fontweight="bold")
            ax.set_ylabel("Nombre de clients")
            ax.set_xlabel("")
            ax.legend(["Fidele", "Resilie"], title="Statut")
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig)
            plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        # Distribution CSAT par churn
        if "csat_score" in data.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            data[data["churn"] == 0]["csat_score"].hist(ax=ax, alpha=0.7, color="#1e3a5f", bins=20, label="Fideles")
            data[data["churn"] == 1]["csat_score"].hist(ax=ax, alpha=0.7, color="#c0392b", bins=20, label="Resilies")
            ax.set_title("Distribution du Score CSAT", fontweight="bold")
            ax.set_xlabel("Score CSAT")
            ax.set_ylabel("Nombre de clients")
            ax.legend()
            st.pyplot(fig)
            plt.close()

    with col_d:
        # Anciennete par churn
        if "tenure_months" in data.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="churn", y="tenure_months", data=data,
                        palette=["#1e3a5f", "#c0392b"], ax=ax)
            ax.set_title("Anciennete selon le Churn", fontweight="bold")
            ax.set_xlabel("Churn (0 = Fidele, 1 = Resilie)")
            ax.set_ylabel("Anciennete (mois)")
            st.pyplot(fig)
            plt.close()


# ==============================================================
# TAB 2 — ANALYSE DES RISQUES
# ==============================================================
with tab2:
    st.markdown('<div class="section-header">Analyse des Clients a Risque dans le Dataset</div>', unsafe_allow_html=True)
    st.info("Cette section applique le modele sur l'ensemble du dataset pour identifier les profils a risque.")

    @st.cache_data
    def compute_risk_scores(_model, _columns, data):
        df_clean = data.drop(columns=["customer_id"], errors="ignore")
        df_enc = pd.get_dummies(df_clean, drop_first=False)
        # remove churn if present
        df_enc = df_enc.drop(columns=["churn"], errors="ignore")
        aligned = pd.DataFrame(0, index=df_enc.index, columns=_columns)
        for col in _columns:
            if col in df_enc.columns:
                aligned[col] = df_enc[col].values
        probas = _model.predict_proba(aligned)[:, 1]
        return probas

    with st.spinner("Calcul des scores de risque sur le dataset complet..."):
        risk_scores = compute_risk_scores(model, expected_columns, data)

    data_risk = data.copy()
    data_risk["churn_proba"] = risk_scores
    data_risk["risk_level"] = pd.cut(
        risk_scores,
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Faible", "Moyen", "Eleve"]
    )

    col_e, col_f = st.columns(2)

    with col_e:
        # Distribution des scores de risque
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(risk_scores, bins=30, color="#5c6bc0", edgecolor="white", alpha=0.85)
        ax.axvline(x=0.5, color="#c0392b", linestyle="--", linewidth=1.5, label="Seuil 0.5")
        ax.set_title("Distribution des Probabilites de Churn\n(modele sur tout le dataset)", fontweight="bold")
        ax.set_xlabel("Probabilite de churn")
        ax.set_ylabel("Nombre de clients")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_f:
        # Repartition par niveau de risque
        risk_counts = data_risk["risk_level"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_risk = {"Faible": "#388e3c", "Moyen": "#f57c00", "Eleve": "#c0392b"}
        bars = ax.bar(risk_counts.index, risk_counts.values,
                      color=[colors_risk.get(k, "grey") for k in risk_counts.index],
                      edgecolor="white", linewidth=1.5)
        ax.set_title("Repartition par Niveau de Risque", fontweight="bold")
        ax.set_ylabel("Nombre de clients")
        for bar, val in zip(bars, risk_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 20,
                    str(val), ha="center", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Top 20 Clients a Risque Eleve</div>', unsafe_allow_html=True)

    display_cols = ["customer_id"] if "customer_id" in data_risk.columns else []
    for col in ["contract_type", "tenure_months", "monthly_fee", "csat_score",
                "payment_failures", "support_tickets", "churn_proba"]:
        if col in data_risk.columns:
            display_cols.append(col)

    top_risk = data_risk.sort_values("churn_proba", ascending=False).head(20)[display_cols]
    top_risk["churn_proba"] = top_risk["churn_proba"].map(lambda x: f"{x*100:.1f}%")
    st.dataframe(top_risk, use_container_width=True)

    # Importance des variables
    st.markdown('<div class="section-header">Variables les Plus Influentes (Feature Importance XGBoost)</div>', unsafe_allow_html=True)
    xgb_clf = model.named_steps["classifier"]
    importances = pd.Series(xgb_clf.feature_importances_, index=expected_columns)
    top15 = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_feat = ["#c0392b" if i >= len(top15) - 3 else "#5c6bc0" for i in range(len(top15))]
    top15.plot(kind="barh", ax=ax, color=colors_feat)
    ax.set_title("Top 15 Variables les Plus Importantes pour Predire le Churn", fontweight="bold")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close()


# ==============================================================
# TAB 3 — SIMULATEUR CLIENT
# ==============================================================
with tab3:
    st.markdown('<div class="section-header">Simulation du Risque d\'un Client</div>', unsafe_allow_html=True)
    st.markdown("Renseignez les caracteristiques d'un client pour obtenir sa probabilite de resiliation en temps reel.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Profil**")
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Genre", ["Female", "Male"])
        country = st.selectbox("Pays", ["USA", "UK", "Canada", "France", "Germany"])
        customer_segment = st.selectbox("Segment", ["Individual", "SMB", "Enterprise"])
        contract_type = st.selectbox("Type de contrat", ["Monthly", "Yearly", "Quarterly"])

    with col2:
        st.markdown("**Utilisation & Engagement**")
        tenure_months = st.slider("Anciennete (mois)", 0, 72, 12)
        monthly_fee = st.number_input("Frais mensuels ($)", 10, 500, 50)
        monthly_logins = st.slider("Connexions par mois", 0, 100, 15)
        weekly_active_days = st.slider("Jours actifs / semaine", 0, 7, 3)
        avg_session_time = st.slider("Duree session moyenne (min)", 1, 120, 15)
        usage_growth_rate = st.slider("Croissance utilisation", -1.0, 1.0, 0.0, step=0.05)

    with col3:
        st.markdown("**Support & Satisfaction**")
        csat_score = st.slider("Score CSAT", 1.0, 5.0, 4.0, step=0.5)
        nps_score = st.slider("Score NPS", 0, 100, 50)
        support_tickets = st.slider("Tickets support", 0, 10, 1)
        payment_failures = st.slider("Echecs de paiement", 0, 5, 0)
        escalations = st.slider("Escalades", 0, 5, 0)

    client_features = {
        "age": age,
        "tenure_months": tenure_months,
        "monthly_fee": monthly_fee,
        "total_revenue": tenure_months * monthly_fee,
        "support_tickets": support_tickets,
        "csat_score": csat_score,
        "contract_type": contract_type,
        "payment_failures": payment_failures,
        "usage_growth_rate": usage_growth_rate,
        "gender": gender,
        "country": country,
        "city": "Paris",
        "customer_segment": customer_segment,
        "signup_channel": "Web",
        "monthly_logins": monthly_logins,
        "weekly_active_days": weekly_active_days,
        "avg_session_time": avg_session_time,
        "features_used": 3,
        "last_login_days_ago": 2,
        "payment_method": "Card",
        "discount_applied": "No",
        "price_increase_last_3m": "No",
        "avg_resolution_time": 24.0,
        "complaint_type": "None",
        "escalations": escalations,
        "email_open_rate": 0.5,
        "marketing_click_rate": 0.1,
        "nps_score": nps_score,
        "survey_response": "Neutral",
        "referral_count": 0
    }

    st.divider()

    result, source = get_prediction(client_features)
    churn_prob = result["churn_probability"]
    revenue_client = tenure_months * monthly_fee

    st.markdown(f"*Source de la prediction : {source}*")

    r1, r2, r3 = st.columns(3)
    with r1:
        color = "red" if churn_prob > 0.5 else "green"
        level = "RISQUE ELEVE" if churn_prob > 0.5 else "CLIENT FIDELE"
        st.markdown(f"""
        <div class="{'kpi-box-danger' if churn_prob > 0.5 else 'kpi-box'}">
            <div class="kpi-title">Statut du Client</div>
            <div class="kpi-value">{level}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="{'kpi-box-danger' if churn_prob > 0.5 else 'kpi-box'}">
            <div class="kpi-title">Probabilite de Resiliation</div>
            <div class="kpi-value">{churn_prob*100:.1f} %</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="kpi-box-danger">
            <div class="kpi-title">Revenu a Risque (ce client)</div>
            <div class="kpi-value">$ {revenue_client * churn_prob:.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.progress(float(churn_prob))

    if churn_prob > 0.5:
        st.error(
            "Ce client presente un risque de resiliation eleve. "
            "Actions recommandees : offre promotionnelle, appel de fidelisation, audit de satisfaction."
        )
    else:
        st.success(
            "Ce client est actuellement fidele. "
            "Maintenez l'engagement via des communications regulieres."
        )

    # Visualisation des facteurs
    st.markdown('<div class="section-header">Facteurs Cles pour ce Client</div>', unsafe_allow_html=True)
    factors = {
        "Echecs paiement": payment_failures / 5,
        "Satisfaction (inverse)": 1 - (csat_score - 1) / 4,
        "Anciennete (protectrice)": 1 - min(tenure_months / 72, 1),
        "Tickets support": support_tickets / 10,
        "Croissance (inverse)": max(0, -usage_growth_rate),
    }
    fig, ax = plt.subplots(figsize=(8, 3))
    colors_f = ["#c0392b" if v > 0.5 else "#388e3c" for v in factors.values()]
    ax.barh(list(factors.keys()), list(factors.values()), color=colors_f)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Indicateurs de risque pour ce client (rouge = signal negatif)", fontsize=10)
    st.pyplot(fig)
    plt.close()


# ==============================================================
# TAB 4 — PERFORMANCE DES MODELES
# ==============================================================
with tab4:
    st.markdown('<div class="section-header">Comparaison des Modeles ML / Deep Learning</div>', unsafe_allow_html=True)

    if comparison_results is None:
        st.warning("Fichier models/comparison_results.json introuvable. Executez le notebook 02 pour generer les comparaisons.")
        st.stop()

    metrics_df = pd.DataFrame(comparison_results["metrics"]).T
    cv_df = pd.DataFrame(comparison_results["cross_validation"]).T

    st.markdown("**Metriques sur le jeu de test (20%)**")
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color="#c8e6c9").format("{:.4f}"),
        use_container_width=True
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig, ax = plt.subplots(figsize=(7, 5))
        metrics_df[["Accuracy", "F1 Score", "ROC AUC"]].plot(
            kind="bar", ax=ax, colormap="Set2", edgecolor="white"
        )
        ax.set_title("Comparaison des Performances", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close()

    with col_m2:
        fig, ax = plt.subplots(figsize=(7, 5))
        auc_vals = cv_df["cv_auc_mean"]
        auc_stds = cv_df["cv_auc_std"]
        best_name = auc_vals.idxmax()
        bar_colors = ["#c0392b" if n == best_name else "#90a4ae" for n in auc_vals.index]
        bars = ax.bar(auc_vals.index, auc_vals.values, yerr=auc_stds.values,
                      capsize=5, color=bar_colors, edgecolor="white")
        ax.set_title("Validation Croisee 5-Fold — ROC AUC\n(moyenne +/- ecart-type)", fontweight="bold")
        ax.set_ylabel("ROC AUC")
        ax.set_ylim(0.4, 1.0)
        ax.tick_params(axis="x", rotation=20)
        for bar, val, std in zip(bars, auc_vals.values, auc_stds.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Precision vs Recall — Dilemme Business</div>', unsafe_allow_html=True)
    st.markdown(
        "Le Recall (taux de detection des churners reels) est plus critique que la Precision dans ce contexte : "
        "**manquer un client a risque coute plus cher** que contacter un client fidele par erreur."
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(metrics_df))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], metrics_df["Precision"], width,
                   label="Precision", color="#1e3a5f", edgecolor="white")
    bars2 = ax.bar([i + width/2 for i in x], metrics_df["Recall"], width,
                   label="Recall", color="#c0392b", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df.index, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Precision vs Recall par Modele (classe Churn)", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.0)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", fontsize=8)
    st.pyplot(fig)
    plt.close()

    best_model = metrics_df["ROC AUC"].idxmax()
    best_cv = cv_df["cv_auc_mean"].idxmax()
    st.markdown(f"""
    **Modele selectionne pour la production : {best_model}**
    - Meilleur ROC AUC sur le jeu de test : {metrics_df.loc[best_model, 'ROC AUC']:.4f}
    - Cross-validation stable : {cv_df.loc[best_cv, 'cv_auc_mean']:.4f} +/- {cv_df.loc[best_cv, 'cv_auc_std']:.4f}
    - Bon compromis performance / interpretabilite / facilite de deploiement
    """)
