#!/bin/sh
set -e

echo "=========================================="
echo " [1/2] Lancement du scraper..."
echo "=========================================="
python scraper/scraper.py

echo "=========================================="
echo " [2/2] Lancement du nettoyage des données..."
echo "=========================================="
python pipeline/clean.py

echo "=========================================="
echo " Pipeline terminé avec succès."
echo "=========================================="
