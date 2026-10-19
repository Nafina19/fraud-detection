# Détection de fraude par carte bancaire

Projet de data science : repérer les rares transactions frauduleuses d'un jeu de données de cartes bancaires, sans submerger les analystes de fausses alertes.

## Le problème

Sur 284 807 transactions, seulement **492 sont des fraudes (0,17 %)**. Un modèle qui répondrait toujours « normale » aurait 99,8 % de bonnes réponses sans détecter aucune fraude : l'accuracy est donc inutilisable. Le projet s'appuie sur la **précision**, le **rappel** et l'**Average Precision (PR-AUC)**.

## Les données

- Source : jeu de données public « Credit Card Fraud Detection » (transactions de cartes européennes, deux jours de septembre 2013), non inclus dans ce dépôt.
- `Time` (secondes depuis la première transaction), `Amount`, `V1` à `V28` (composantes d'ACP anonymisées) et `Class` (1 = fraude).
- Aucune valeur manquante ; 1 081 doublons exacts, conservés puis étudiés à part.

## Méthode

1. **Exploration (EDA)** : qualité des données, déséquilibre, `Amount`, `Time`, variables `V` (AUC univariée), intervalles de confiance.
2. **Préparation** : `Amount_log` (log1p), heure encodée de façon cyclique, `V1` à `V28` ; `Time` brut retiré.
3. **Découpage** stratifié 60 % entraînement / 20 % validation / 20 % test.
4. **Modèles** : baseline, régression logistique, Random Forest, XGBoost, comparés par Average Precision et validation croisée.
5. **Seuil de décision** choisi sur la validation, par le F1 puis par un coût métier (montants des fraudes manquées + coût d'examen des alertes).
6. **Contrôles** : sensibilité aux doublons, importance par permutation, analyse des fraudes manquées.

## Résultats principaux (jeu de test, 98 fraudes)

| Modèle | AP validation | AP validation croisée | AP test |
|---|---|---|---|
| XGBoost (retenu) | 0,815 | 0,837 ± 0,026 | 0,856 |
| Random Forest | 0,810 | 0,843 ± 0,022 | 0,864 |
| Régression logistique | 0,669 | 0,755 ± 0,037 | 0,707 |
| Baseline (hasard) | 0,002 | | 0,002 |

Pour XGBoost, au seuil maximisant le F1 sur la validation (0,982) : **76 fraudes détectées sur 98, 10 fausses alertes** (précision 88 %, rappel 78 %).

- Avec une hypothèse de coût de 5 par alerte, le modèle réduit les pertes d'environ **75 %** par rapport à l'absence de modèle ; ce chiffre dépend fortement de cette hypothèse (81 % pour 1, 42 % pour 20).
- XGBoost et Random Forest sont **indiscernables** compte tenu de la variabilité observée.
- Les variables les plus importantes sont V4, V14, V12, V10 et V11.
- Les doublons gonflent très légèrement les scores (environ 0,014 d'AP) sans changer les conclusions.

## Limites

- Seulement 492 fraudes et deux jours de données : les estimations et les seuils sont incertains.
- Variables anonymisées : on mesure leur importance, pas leur signification.
- L'heure suppose que `Time = 0` correspond à minuit ; l'effet de l'heure est instable d'un jour à l'autre (seule l'heure 2 est élevée les deux jours).
- Le coût d'une alerte est une hypothèse.
- Hyperparamètres non optimisés ; découpage aléatoire plutôt que chronologique.

## Structure du dépôt

```
├── data/
│   ├── raw/            creditcard.csv (à télécharger)
│   └── processed/      df_enrichi.csv, fraud_powerbi.csv
├── models/             fraud_model.joblib
├── notebooks/
│   ├── 01_eda_final.ipynb
│   └── 02_modelisation_final.ipynb
├── README.md
└── requirements.txt
```

## Reproduire

```bash
pip install -r requirements.txt
```

Placer `creditcard.csv` dans `data/raw/`, puis exécuter dans l'ordre `01_eda_final.ipynb` (qui crée `df_enrichi.csv`) et `02_modelisation_final.ipynb` (le Random Forest et la validation croisée prennent plusieurs minutes ; `RUN_CV = False` accélère l'exécution).

## Suite possible

Réglage des hyperparamètres, choix du seuil par validation croisée, calibration des probabilités, explications individuelles (SHAP), tableau de bord Power BI à partir de `fraud_powerbi.csv`.
