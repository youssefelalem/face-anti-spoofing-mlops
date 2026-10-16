"""Configuration centrale du projet (chemins, hyperparametres, constantes).

Toute valeur partagee entre l'entrainement, l'evaluation et l'application
doit etre definie ICI, pour eviter les ecarts entrainement / production
(training-serving skew).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Donnees ---------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
NUAA_DIR = DATA_RAW / "nuaa" / "Detectedface"
NUAA_ZIP = DATA_RAW / "_downloads" / "nuaa_detectedface.zip"
DATA_PROCESSED = ROOT / "data" / "processed"
MANIFEST_CSV = DATA_PROCESSED / "manifest.csv"
DATA_CONTRACT = ROOT / "data_contract.yaml"

# Donnees webcam propres au projet (app/collect.py) : domaine de l'application reelle.
WEBCAM_DIR = DATA_RAW / "webcam"
WEBCAM_MANIFEST_CSV = DATA_PROCESSED / "webcam_manifest.csv"
WEBCAM_TEST_SESSIONS = ("s2",)     # sessions reservees au test (jamais vues a l'entrainement)
USE_WEBCAM = True
WEBCAM_OVERSAMPLE = 3              # peu d'images webcam : on les repete dans le jeu d'entrainement

# Protocole de validation :
# - "subject" : sujets VAL_SUBJECTS retires du train (sessions 01-02) -> plus de donnees d'entrainement.
# - "session" : train = session 01, validation = session 02 (utilise par src/experiments.py).
# Aucun des deux ne predit bien la session 03 (voir reports/experiments.md) : les sessions 01-02
# partagent le meme biais (attaques imprimees pales), absent de la session 03.
VAL_PROTOCOL = "subject"
VAL_SUBJECTS = ("0008", "0009")
VAL_SESSION = "02"

# Etiquettes : 0 = visage reel (bona fide), 1 = attaque (spoof)
LABELS = {"real": 0, "spoof": 1}
CLASS_NAMES = ["real", "spoof"]

# --- Pretraitement (identique entrainement / inference) ---------------------
IMG_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Entrainement ------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
HEAD_EPOCHS = 3          # phase 1 : seule la tete de classification est entrainee
FINETUNE_EPOCHS = 12     # phase 2 : fine-tuning de tout le reseau
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 3
# Configuration finale : celle de la v1 (couleur, augmentation de base), la meilleure des
# variantes CNN essayees. La v2 (niveaux de gris + normalisation photometrique + augmentation
# forte) visait les raccourcis detectes par le data contract mais a moins bien generalise
# (voir reports/experiments.md). Les options restent disponibles en ligne de commande.
STRONG_AUG = False
GRAYSCALE = False
PHOTO_NORM = False
FREEZE_BLOCKS = 0

# --- Artefacts ---------------------------------------------------------------
MODELS_DIR = ROOT / "models" / "saved"
CHECKPOINT = MODELS_DIR / "mobilenetv2_fas.pt"
ONNX_MODEL = MODELS_DIR / "mobilenetv2_fas.onnx"
MODEL_META = MODELS_DIR / "model_meta.json"

MLFLOW_DIR = ROOT / "experiments" / "mlflow_runs"
MLFLOW_TRACKING_URI = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (MLFLOW_DIR / "artifacts").as_uri()
MLFLOW_EXPERIMENT = "face-antispoofing-nuaa"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
