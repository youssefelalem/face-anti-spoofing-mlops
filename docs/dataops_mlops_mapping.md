# Mapping cours MLOps / DataOps / DevOps → projet Face Anti-Spoofing

Chaque concept vu en cours, et l'endroit exact où il est appliqué dans ce projet.

## Chapitre 2 — Pipelines agiles & orchestration (Pr. AIT DAOUD, Pr. AMZIL)

| Concept du cours | Application dans le projet |
|---|---|
| Pipeline de données : Ingestion → Validation → Nettoyage → Transformation → Stockage | `data/raw/_downloads/` (ingestion) → `src/preprocessing/build_manifest.py` (indexation) → `src/preprocessing/validate_data.py` (validation) → `src/preprocessing/dataset.py` (transformation, cache) |
| Pipeline ML : Split → Entraînement → Évaluation → Artefact | `build_manifest.py` (split) → `src/train.py` → `src/evaluate.py` → `src/export_onnx.py` (artefact ONNX) |
| Pipeline idempotent, reproductible | graine fixe (`SEED = 42`), split déterministe, empreinte SHA-256 de l'archive dans `data_contract.yaml` |
| Versionnement étendu : code, données, configs, modèles, environnements | code : Git · données : empreinte SHA-256 (archive NUAA) · config : `src/config.py` · modèles : MLflow + `model_meta.json` (SHA-256 de l'ONNX) · environnement : `requirements*.txt` + `Dockerfile` |
| Orchestration (DAG) | chaîne linéaire `build_manifest → validate_data → train → evaluate → export_onnx` ; chaque étape échoue explicitement si la précédente n'a pas produit son artefact |

## Chapitre 3 — Data Quality & Data Contracts

| Concept du cours | Application dans le projet |
|---|---|
| Data Contract (YAML, accord producteur/consommateur) | `data_contract.yaml` : source, licence, schéma du manifeste, règles de qualité, règles anti-fuite |
| Shift Left | la validation s'exécute **avant** l'entraînement ; code de sortie ≠ 0 si le contrat est violé |
| Complétude | `fichiers_presents` : 0 fichier référencé manquant |
| Validité / Domaine | `schema` : label ∈ {0, 1}, session ∈ {01, 02, 03}, sujet au format `NNNN` |
| Unicité | `doublons` : aucun chemin en double |
| Exactitude / Cohérence | `images_decodables`, `canaux_couleur`, `taille_minimale`, `format_carre` |
| Intégrité / Traçabilité | `empreinte_archive` (SHA-256) ; rapport `reports/data_quality_report.json` |
| Cohérence métier | `volumetrie` (effectifs officiels NUAA) et `equilibre_classes_train` |
| Tests de données | `tests/test_data_quality.py` (manifeste, garde-fous anti-fuite) |
| Fuite de données (data leakage) | sessions train/test disjointes, sujets train/val disjoints, aucun fichier dans deux splits |
| Tests d'anomalie / biais (cohérence métier) | `biais_luminance_moyenne` (AUC 0,92) et `biais_saturation_moyenne` (AUC 0,89) : une caractéristique triviale sépare les classes → avertissement non bloquant, qui a expliqué l'échec de généralisation |
| Feedback loop / Kaizen (Mesurer → Détecter → Analyser → Corriger → Réévaluer) | échec sur la session 3 → diagnostic (biais) → essais (niveaux de gris, normalisation) → ajout de la texture LBP et de données webcam → réévaluation sur une session indépendante (`reports/experiments.md`) |
| Données du domaine de production | collecte webcam (`app/collect.py`), nettoyage automatique des faux positifs, découpage temporel train/val, session s2 réservée au test |

## Chapitre 5 — CI/CD, conteneurisation, déploiement

| Concept du cours | Application dans le projet |
|---|---|
| CI : tests + lint à chaque commit | `.github/workflows/ci.yml` : `ruff` + `pytest` à chaque push / pull request |
| Tests fonctionnels d'inférence, tests de contrat | `tests/test_api.py` (routes, formats de réponse, image invalide) avec un modèle ONNX factice |
| Packaging : prototype → package → service | notebook/scripts → modèle ONNX + `model_meta.json` → service FastAPI |
| Docker : image légère, dépendances figées, pas de secrets, commande explicite | `Dockerfile` : `python:3.12-slim`, sans PyTorch (ONNX Runtime seul), utilisateur non-root, `HEALTHCHECK`, `CMD` explicite |
| "Build once, run many" | la même image sert en local et en démonstration |
| Service ML de type API (réponse synchrone) | `POST /predict`, `GET /health`, `GET /model` |
| Smoke test | `HEALTHCHECK` Docker sur `/health` ; job CI qui construit l'image |
| Stratégie de déploiement | **Recreate** (serveur unique, projet académique) |

## Cloud Computing & DevOps (Pr. F. Benabbou)

| Concept du cours | Application dans le projet |
|---|---|
| Git, branche principale stable | dépôt Git, branche `main` |
| CI (équivalent Jenkins) | GitHub Actions |
| Configuration déclarative (esprit IaC) | `Dockerfile`, `ci.yml`, `data_contract.yaml` : tout l'environnement est décrit dans des fichiers versionnés |

## Machine Learning (Pr. BEN LAHMAR EL Habib)

| Concept du cours | Application dans le projet |
|---|---|
| Supervised Learning → Classification binaire | real (0) / spoof (1) |
| Techniques supervisées classiques (SVM) | LBP + SVM RBF (`src/baseline_lbp_svm.py`), comparé puis fusionné avec le CNN |
| Training set / Testing set | train = sessions 01-02 (sujets hors validation), test = session 03 |
| No free lunch : même distribution train / test | limite identifiée : NUAA ne contient que des attaques par photo imprimée → robustesse non garantie face aux attaques par écran ou masque |
| Under-fitting / Over-fitting | validation sur des sujets jamais vus + early stopping + dropout + augmentation de données |
