import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def get_latest_raw_file(raw_dir="data/raw") -> Path:
    files = list(Path(raw_dir).glob("madarent_raw_*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {raw_dir}")
    return max(files, key=lambda f: f.name)


def clean_price(text):
    if pd.isna(text):
        return None
    digits = re.sub(r"[^\d]", "", str(text))
    return int(digits) if digits else None

def clean_pieces(text):
    if pd.isna(text):
        return None
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else None

def to_boolean_presence(text):
    return pd.notna(text)

def extract_toilet_count(text):
    if pd.isna(text):
        return 0
    match = re.search(r"(\d+)\s*sdb", text)
    return int(match.group(1)) if match else 1

def save_clean_data(df: pd.DataFrame, output_dir="data/processed"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = Path(output_dir) / f"madarent_clean_{timestamp}.csv"
    logger.info(f"Chemin du fichier : {filepath}")
    df.to_csv(filepath, index=False)
    logger.info(f"Données sauvegardées")
    return filepath

latest_file = get_latest_raw_file()
logger.info(f"Fichier source : {latest_file}")
df = pd.read_csv(latest_file)

df["pieces"] = df["pieces"].apply(clean_pieces)
logger.info(f"Nettoyage pièce terminé")
df["chambres"] = df["chambres"].apply(clean_pieces)
logger.info(f"Nettoyage chambre terminé")
df["pieces"] = df["pieces"].combine_first(df["chambres"]).fillna(0).astype("Int64")
df = df.drop(columns="chambres")
logger.info(f"Fusion pièce et chambre terminée")

df["cuisine"] = df["cuisine"].apply(to_boolean_presence)
df["Charges"] = df["Charges"].apply(to_boolean_presence)
logger.info(f"Nettoyage cuisine et charges terminé")

df["toilettes_presence"] = df["toilettes"].notna()
df["toilettes_count"] = df["toilettes"].apply(extract_toilet_count)
df = df.drop(columns="toilettes")
logger.info(f"Nettoyage toilettes terminé")

df["Acces moto"] = df["Acces moto"].notna()
df["Acces voiture"] = df["Acces voiture"].notna()
logger.info(f"Nettoyage acces terminé")

df = df.drop(columns="description")
df = df.drop(columns="titre")
logger.info(f"Suppression colonne description et titre terminé")

df = df.dropna(subset=["prix"])
logger.info(f"Suppression ligne avec prix manquant terminé")
df["prix_mensuel"] = np.where(
    df["unité du prix"] == "/jour",
    df["prix"] * 30,
    df["prix"]
)
df = df.drop(columns=["prix", "unité du prix"])
logger.info(f"Conversion prix en mensuel terminé")

# On supprime la colonne car c'est toujours false pour le scraper actuel
df = df.drop(columns="Charges")
logger.info(f"Suppression colonne charges terminé")

save_clean_data(df)