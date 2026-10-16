# Image de l'application : FastAPI + ONNX Runtime (sans PyTorch -> image legere).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Code strictement necessaire a l'inference
COPY src/__init__.py src/__init__.py
COPY src/utils/__init__.py src/utils/preprocess.py src/utils/face_detect.py src/utils/lbp.py src/utils/
COPY app/ app/
# Modeles exportes (ONNX + meta, LBP-SVM + fusion), non versionnes dans Git.
# Les checkpoints PyTorch (.pt) sont exclus par .dockerignore.
COPY models/saved/ models/saved/

# Execution sans privileges root ; le dossier des modeles doit lui appartenir, car
# l'application peut y telecharger les modeles au demarrage (deploiement Render).
RUN useradd --create-home appuser && chown -R appuser /srv/models
USER appuser

EXPOSE 8000
# Le port peut etre impose par l'hebergeur (variable PORT, ex. Render) ; 8000 par defaut.
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
