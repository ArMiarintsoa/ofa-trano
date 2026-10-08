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
    files = list(Path(raw_dir).glob("madarent_*.csv"))
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
    df.to_csv(filepath, index=False)
    return filepath

latest_file = get_latest_raw_file()
lo
df = pd.read_csv(latest_file)

df["pieces"] = df["pieces"].apply(clean_pieces)
df["chambres"] = df["chambres"].apply(clean_pieces)
df["pieces"] = df["pieces"].combine_first(df["chambres"]).fillna(0).astype("Int64")
df = df.drop(columns="chambres", axis=1)

df["cuisine"] = df["cuisine"].apply(to_boolean_presence)
df["Charges"] = df["Charges"].apply(to_boolean_presence)

df["toilettes_presence"] = df["toilettes"].notna()
df["toilettes_count"] = df["toilettes"].apply(extract_toilet_count)
df = df.drop(columns="toilettes", axis=1)

df["Acces moto"] = df["Acces moto"].notna()
df["Acces voiture"] = df["Acces voiture"].notna()

df = df.drop(columns="description", axis=1)
df = df.drop(columns="titre", axis=1)

df = df.dropna(subset=["prix"])
df["prix_mensuel"] = np.where(
    df["unité du prix"] == "/jour",
    df["prix"] * 30,
    df["prix"]
)
df = df.drop(columns=["prix", "unité du prix"], axis=1)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[["ville", "lieu", "type"]])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["ville", "lieu", "type"]))
df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

df = df.drop(columns=["ville", "lieu", "type"])

save_clean_data(df)