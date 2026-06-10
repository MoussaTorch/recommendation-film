"""
data_loader.py
--------------
Pipeline de données : téléchargement (→ raw), nettoyage/filtrage (→ interim).

Étapes :
  1. download_movielens()  : télécharge + extrait dans data/raw/
  2. process_ratings()     : filtre MIN_RATINGS, sauvegarde data/interim/ (.csv)
  3. process_movies()      : nettoie titres/genres,  sauvegarde data/interim/ (.csv)
  4. load_*()              : charge depuis interim (usage courant dans le code)
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings.params import (
    MOVIELENS_URL, MOVIELENS_ZIP, MOVIELENS_DIR,
    RATINGS_FILE, MOVIES_FILE, TAGS_FILE,
    RATINGS_INTERIM, MOVIES_INTERIM, TAGS_INTERIM,
    DATA_DIR_RAW, MODEL_PARAMS,
)

log_fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss!UTC}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"
)
logger.remove()
logger.add(sys.stderr, format=log_fmt, level="INFO")


# ── 1. Téléchargement (→ raw) ─────────────────────────────────────────────────
def download_movielens(force: bool = False) -> None:
    """Télécharge et extrait MovieLens ml-latest-small dans data/raw/."""
    if MOVIELENS_DIR.exists() and not force:
        logger.info(f"Dataset brut déjà présent : {MOVIELENS_DIR}")
        return

    logger.info(f"Téléchargement depuis {MOVIELENS_URL}")
    urllib.request.urlretrieve(MOVIELENS_URL, MOVIELENS_ZIP)
    logger.success(f"Téléchargé : {MOVIELENS_ZIP}")

    logger.info("Extraction ZIP → data/raw/")
    with zipfile.ZipFile(MOVIELENS_ZIP, "r") as zf:
        zf.extractall(DATA_DIR_RAW)
    logger.success(f"Extrait : {MOVIELENS_DIR}")


# ── 2. Traitement ratings (raw → interim) ─────────────────────────────────────
def process_ratings(min_ratings: int = MODEL_PARAMS["MIN_RATINGS"],
                    force: bool = False) -> pd.DataFrame:
    """
    Lit ratings.csv (raw), applique le filtre MIN_RATINGS,
    sauvegarde ratings_filtered.csv dans data/interim/.

    Filtre double : on retire les utilisateurs ET les films ayant moins
    de `min_ratings` interactions (cold-start mitigation).
    """
    if RATINGS_INTERIM.exists() and not force:
        logger.info(f"Interim ratings déjà présent : {RATINGS_INTERIM}")
        return pd.read_csv(RATINGS_INTERIM)

    logger.info(f"Chargement raw ratings : {RATINGS_FILE}")
    df = pd.read_csv(RATINGS_FILE)
    df.columns = df.columns.str.lower()

    n_raw = len(df)
    logger.info(
        f"  Brut : {n_raw:,} notes | "
        f"{df['userid'].nunique():,} users | "
        f"{df['movieid'].nunique():,} films"
    )

    user_counts = df["userid"].value_counts()
    item_counts = df["movieid"].value_counts()
    active_users = user_counts[user_counts >= min_ratings].index
    active_items = item_counts[item_counts >= min_ratings].index

    df = df[
        df["userid"].isin(active_users) & df["movieid"].isin(active_items)
    ].copy()

    n_filtered = len(df)
    logger.info(
        f"  Après filtre ≥{min_ratings} notes : {n_filtered:,} notes "
        f"({n_filtered / n_raw * 100:.1f}% conservées) | "
        f"{df['userid'].nunique():,} users | "
        f"{df['movieid'].nunique():,} films"
    )

    df.to_csv(RATINGS_INTERIM, index=False)
    logger.success(f"Sauvegardé : {RATINGS_INTERIM}")
    return df


# ── 3. Traitement films (raw → interim) ───────────────────────────────────────
def process_movies(force: bool = False) -> pd.DataFrame:
    """
    Lit movies.csv (raw), extrait l'année et nettoie le titre,
    sauvegarde movies_clean.csv dans data/interim/.

    genres_list est sauvegardé en string pipe-separated (ex: "Action|Comedy")
    et reconverti en liste Python au chargement.
    """
    if MOVIES_INTERIM.exists() and not force:
        logger.info(f"Interim movies déjà présent : {MOVIES_INTERIM}")
        df = pd.read_csv(MOVIES_INTERIM)
        # Reconvertit la colonne pipe-separated en liste Python
        df["genres_list"] = df["genres_list"].str.split("|")
        return df

    logger.info(f"Chargement raw movies : {MOVIES_FILE}")
    df = pd.read_csv(MOVIES_FILE)
    df.columns = df.columns.str.lower()

    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("float")
    df["title_clean"] = (
        df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    )
    df["genres_list"] = df["genres"].str.split("|")

    logger.info(f"  {len(df):,} films | années : {df['year'].min():.0f}–{df['year'].max():.0f}")

    # Sauvegarder genres_list en string pipe-separated pour la compatibilité CSV
    df_save = df.copy()
    df_save["genres_list"] = df_save["genres_list"].apply("|".join)
    df_save.to_csv(MOVIES_INTERIM, index=False)
    logger.success(f"Sauvegardé : {MOVIES_INTERIM}")
    return df


def process_tags(force: bool = False) -> pd.DataFrame:
    """Lit tags.csv (raw) et sauvegarde tags_clean.csv dans data/interim/."""
    if TAGS_INTERIM.exists() and not force:
        logger.info(f"Interim tags déjà présent : {TAGS_INTERIM}")
        return pd.read_csv(TAGS_INTERIM)

    logger.info(f"Chargement raw tags : {TAGS_FILE}")
    df = pd.read_csv(TAGS_FILE)
    df.columns = df.columns.str.lower()
    df.to_csv(TAGS_INTERIM, index=False)
    logger.success(f"Sauvegardé : {TAGS_INTERIM}")
    return df


# ── 4. Chargement depuis interim (usage courant) ──────────────────────────────
def load_ratings(min_ratings: int = MODEL_PARAMS["MIN_RATINGS"]) -> pd.DataFrame:
    """Charge ratings filtrés depuis interim (lance process_ratings si nécessaire)."""
    return process_ratings(min_ratings=min_ratings)


def load_movies() -> pd.DataFrame:
    """Charge movies nettoyés depuis interim (lance process_movies si nécessaire)."""
    return process_movies()


def load_tags() -> pd.DataFrame:
    return process_tags()


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "ratings": load_ratings(),
        "movies" : load_movies(),
        "tags"   : load_tags(),
    }


# ── 5. Métriques descriptives ─────────────────────────────────────────────────
def dataset_summary(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """Retourne un dict de métriques descriptives (loggable via MLflow)."""
    n_ratings = len(ratings)
    n_users   = ratings["userid"].nunique()
    n_items   = ratings["movieid"].nunique()
    density   = n_ratings / (n_users * n_items)

    summary = {
        "n_ratings"  : n_ratings,
        "n_users"    : n_users,
        "n_items"    : n_items,
        "density_pct": round(density * 100, 4),
        "avg_rating" : round(ratings["rating"].mean(), 4),
        "std_rating" : round(ratings["rating"].std(), 4),
        "n_genres"   : movies["genres_list"].explode().nunique(),
    }
    logger.info(f"Résumé dataset : {summary}")
    return summary


# ── Pipeline complet ──────────────────────────────────────────────────────────
def run_data_pipeline(force: bool = False) -> dict[str, pd.DataFrame]:
    """Lance le pipeline complet : raw (téléchargement) → interim (nettoyage)."""
    download_movielens(force=force)
    data = {
        "ratings": process_ratings(force=force),
        "movies" : process_movies(force=force),
        "tags"   : process_tags(force=force),
    }
    summary = dataset_summary(data["ratings"], data["movies"])
    logger.success(
        f"Pipeline data terminé — {summary['n_ratings']:,} notes, "
        f"{summary['n_users']} users, {summary['n_items']} films, "
        f"densité {summary['density_pct']:.2f}%"
    )
    return data


if __name__ == "__main__":
    run_data_pipeline()
