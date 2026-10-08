# Image de base Python optimisée
FROM python:3.11-slim

# Empêcher Python de générer des fichiers .pyc et forcer l'affichage immédiat des logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installation de Playwright et des dépendances de navigateurs (Chromium)
RUN playwright install --with-deps chromium

# Copie du reste de l'application
COPY . .

# Rendre le script d'entrée exécutable
RUN chmod +x entrypoint.sh

# Création du dossier de destination des données si inexistant
RUN mkdir -p data/raw data/processed

# Commande par défaut : scraper → nettoyage
CMD ["sh", "entrypoint.sh"]
