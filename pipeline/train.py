import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_latest_clean_file(processed_dir="data/processed") -> Path:
    """Récupère le dernier fichier nettoyé dans data/processed."""
    files = list(Path(processed_dir).glob("madarent_clean_*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier nettoyé trouvé dans {processed_dir}")
    return max(files, key=lambda f: f.name)


def load_and_prepare_data(file_path: Path):
    """Charge les données et sépare les features de la cible."""
    logger.info(f"Chargement des données depuis : {file_path}")
    df = pd.read_csv(file_path)

    target_col = "prix_mensuel"
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' absente du dataset")

    # Nettoyage des valeurs manquantes sur le target
    df = df.dropna(subset=[target_col])

    # Variables
    cat_features = ["type", "lieu"]  # 'ville' est ignorée car constante (1 seule ville)
    num_features = ["pieces", "toilettes_count"]
    bool_features = ["cuisine", "toilettes_presence", "Acces moto", "Acces voiture"]

    # Convertir les booléens en entiers 0/1
    for col in bool_features:
        if col in df.columns:
            df[col] = df[col].astype(int)

    feature_cols = cat_features + num_features + bool_features
    # Ne garder que les colonnes existantes
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Remplir les valeurs manquantes de 'lieu' et 'type' par 'Inconnu'
    if "lieu" in X.columns:
        X["lieu"] = X["lieu"].fillna("Inconnu").astype(str)
    if "type" in X.columns:
        X["type"] = X["type"].fillna("Inconnu").astype(str)

    return X, y, cat_features, num_features + bool_features


def build_pipeline_onehot(cat_features: list[str], other_features: list[str]) -> TransformedTargetRegressor:
    """
    Approche 1: One-Hot Encoding avec RandomForest.
    Intègre une transformation logarithmique log1p sur le prix pour atténuer l'asymétrie.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
            (
                "other",
                SimpleImputer(strategy="median"),
                other_features,
            ),
        ],
        remainder="drop",
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    # Entraîne le modèle sur log(1 + y) et reconvertit automatiquement les prédictions en Ariary
    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return model


def build_pipeline_native_categorical(
    cat_features: list[str], other_features: list[str]
) -> TransformedTargetRegressor:
    """
    Approche 2: HistGradientBoosting avec support catégoriel natif.
    Intègre également une transformation logarithmique log1p sur le prix.
    """
    cat_indices = list(range(len(cat_features)))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                cat_features,
            ),
            (
                "other",
                SimpleImputer(strategy="median"),
                other_features,
            ),
        ],
        remainder="drop",
    )

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    categorical_features=cat_indices,
                    random_state=42,
                ),
            ),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return model


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test, cv_folds=5):
    """Évalue un modèle via Cross-Validation et Test set, et affiche les métriques."""
    logger.info(f"--- Évaluation : {name} ---")

    # 1. Validation croisée
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

    cv_mae = -cv_results["test_mae"].mean()
    cv_rmse = -cv_results["test_rmse"].mean()
    cv_r2 = cv_results["test_r2"].mean()

    # 2. Entraînement sur tout le train set et test final
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    metrics = {
        "name": name,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
    }

    logger.info(
        f"[{name}] CV (5-fold) -> MAE: {cv_mae:,.0f} Ar | RMSE: {cv_rmse:,.0f} Ar | R2: {cv_r2:.4f}"
    )
    logger.info(
        f"[{name}] Test set   -> MAE: {test_mae:,.0f} Ar | RMSE: {test_rmse:,.0f} Ar | R2: {test_r2:.4f}"
    )
    return metrics, model


def save_best_model(model, model_dir="models") -> Path:
    """Sauvegarde le modèle complet entraîné."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(model_dir) / "best_rent_model.pkl"
    joblib.dump(model, out_path)
    logger.info(f"Modèle sauvegardé avec succès dans : {out_path}")
    return out_path


def main():
    logger.info("Démarrage de l'entraînement et comparaison des modèles (avec Log-Target)...")
    clean_file = get_latest_clean_file()
    X, y, cat_features, other_features = load_and_prepare_data(clean_file)

    logger.info(f"Dataset : {X.shape[0]} lignes, {X.shape[1]} features")
    logger.info(f"Catégories : {cat_features} | Autres features : {other_features}")

    # Séparation Train / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Modèle 1 : One-Hot Encoding + RandomForest (avec Log-Target)
    pipe_onehot = build_pipeline_onehot(cat_features, other_features)
    metrics_onehot, fitted_onehot = evaluate_model(
        "One-Hot Encoding + Random Forest (Log-Target)",
        pipe_onehot,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 2. Modèle 2 : Support catégoriel natif + HistGradientBoosting (avec Log-Target)
    pipe_native = build_pipeline_native_categorical(cat_features, other_features)
    metrics_native, fitted_native = evaluate_model(
        "Catégoriel Natif + HistGradientBoosting (Log-Target)",
        pipe_native,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Tableau comparatif
    print("\n" + "=" * 80)
    print("           COMPARAISON DES PERFORMANCES DU MODÈLE (CIBLE LOG)")
    print("=" * 80)
    comparison_df = pd.DataFrame(
        [
            {
                "Modèle": metrics_onehot["name"],
                "CV MAE (Ar)": f"{metrics_onehot['cv_mae']:,.0f}",
                "Test MAE (Ar)": f"{metrics_onehot['test_mae']:,.0f}",
                "Test RMSE (Ar)": f"{metrics_onehot['test_rmse']:,.0f}",
                "Test R²": f"{metrics_onehot['test_r2']:.4f}",
            },
            {
                "Modèle": metrics_native["name"],
                "CV MAE (Ar)": f"{metrics_native['cv_mae']:,.0f}",
                "Test MAE (Ar)": f"{metrics_native['test_mae']:,.0f}",
                "Test RMSE (Ar)": f"{metrics_native['test_rmse']:,.0f}",
                "Test R²": f"{metrics_native['test_r2']:.4f}",
            },
        ]
    )
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Choix et sauvegarde du meilleur modèle basé sur le test MAE
    if metrics_native["test_mae"] < metrics_onehot["test_mae"]:
        best_name = metrics_native["name"]
        best_model = fitted_native
    else:
        best_name = metrics_onehot["name"]
        best_model = fitted_onehot

    logger.info(f"Meilleur modèle sélectionné : {best_name}")
    save_best_model(best_model)


if __name__ == "__main__":
    main()
