"""
test_pipeline.py
----------------
18 tests unitaires pour valider chaque étape du pipeline MLOps.
Lance avec : pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings.params import (
    MODEL_PARAMS, SEED,
    RATINGS_INTERIM, MOVIES_INTERIM,
    DATA_DIR_RAW, DATA_DIR_INTERIM, DATA_DIR_PROCESSED,
)
from src.data_loader import load_ratings, load_movies, dataset_summary
from src.features import (
    build_surprise_dataset,
    build_user_item_matrix,
    build_user_features,
    build_item_features,
    encode_genres,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ratings_df():
    return load_ratings()


@pytest.fixture(scope="module")
def movies_df():
    return load_movies()


@pytest.fixture(scope="module")
def surprise_data(ratings_df):
    return build_surprise_dataset(ratings_df)


# ── Tests data architecture ───────────────────────────────────────────────────
class TestDataArchitecture:

    def test_raw_dir_exists(self):
        assert DATA_DIR_RAW.exists(), "data/raw/ doit exister"

    def test_interim_dir_exists(self):
        assert DATA_DIR_INTERIM.exists(), "data/interim/ doit exister"

    def test_processed_dir_exists(self):
        assert DATA_DIR_PROCESSED.exists(), "data/processed/ doit exister"

    def test_interim_ratings_parquet(self):
        assert RATINGS_INTERIM.exists(), \
            f"Fichier interim manquant : {RATINGS_INTERIM}"

    def test_interim_movies_parquet(self):
        assert MOVIES_INTERIM.exists(), \
            f"Fichier interim manquant : {MOVIES_INTERIM}"


# ── Tests data_loader ─────────────────────────────────────────────────────────
class TestDataLoader:

    def test_ratings_columns(self, ratings_df):
        expected = {"userid", "movieid", "rating", "timestamp"}
        assert expected.issubset(set(ratings_df.columns)), \
            f"Colonnes manquantes : {expected - set(ratings_df.columns)}"

    def test_ratings_not_empty(self, ratings_df):
        assert len(ratings_df) > 0

    def test_rating_scale(self, ratings_df):
        low, high = MODEL_PARAMS["SCALE"]
        assert ratings_df["rating"].between(low, high).all()

    def test_no_duplicate_ratings(self, ratings_df):
        dupes = ratings_df.duplicated(["userid", "movieid"]).sum()
        assert dupes == 0, f"{dupes} notes dupliquées (userid, movieid)"

    def test_movies_columns(self, movies_df):
        expected = {"movieid", "title", "genres"}
        assert expected.issubset(set(movies_df.columns))

    def test_movies_genres_list(self, movies_df):
        assert "genres_list" in movies_df.columns
        assert movies_df["genres_list"].apply(lambda x: isinstance(x, list)).all()

    def test_movies_year_extracted(self, movies_df):
        assert "year" in movies_df.columns
        non_null_years = movies_df["year"].dropna()
        assert (non_null_years >= 1900).all() and (non_null_years <= 2030).all()

    def test_dataset_summary_keys(self, ratings_df, movies_df):
        summary = dataset_summary(ratings_df, movies_df)
        for key in ["n_ratings", "n_users", "n_items", "density_pct", "avg_rating"]:
            assert key in summary, f"Clé manquante : {key}"

    def test_min_ratings_filter(self, ratings_df):
        user_counts = ratings_df.groupby("userid").size()
        assert (user_counts >= MODEL_PARAMS["MIN_RATINGS"]).all()

    def test_real_dataset_size(self, ratings_df):
        """Le vrai dataset doit avoir plus de 80 000 notes (pas des données synthétiques)."""
        assert len(ratings_df) > 80_000, \
            f"Dataset trop petit ({len(ratings_df)}). Relancer data_loader.py pour télécharger le vrai dataset."


# ── Tests features ────────────────────────────────────────────────────────────
class TestFeatures:

    def test_surprise_dataset(self, surprise_data):
        from surprise import Dataset
        assert isinstance(surprise_data, Dataset)

    def test_user_item_matrix_shape(self, ratings_df):
        matrix = build_user_item_matrix(ratings_df, save=False)
        assert matrix.shape[0] == ratings_df["userid"].nunique()
        assert matrix.shape[1] == ratings_df["movieid"].nunique()

    def test_user_features_shape(self, ratings_df):
        uf = build_user_features(ratings_df, save=False)
        assert len(uf) == ratings_df["userid"].nunique()
        assert "user_n_ratings" in uf.columns
        assert "user_mean_rating" in uf.columns

    def test_item_features_shape(self, ratings_df, movies_df):
        itf = build_item_features(ratings_df, movies_df, save=False)
        assert len(itf) == ratings_df["movieid"].nunique()
        assert "item_mean_rating" in itf.columns

    def test_genre_encoding(self, movies_df):
        ge = encode_genres(movies_df, save=False)
        assert "movieid" in ge.columns
        assert len(ge) == len(movies_df)
        genre_cols = [c for c in ge.columns if c not in ["movieid", "title_clean"]]
        assert ge[genre_cols].isin([0, 1]).all().all()


# ── Tests trainer ─────────────────────────────────────────────────────────────
class TestTrainer:

    def test_evaluate_model_keys(self, surprise_data):
        from surprise import NormalPredictor
        from src.trainer import evaluate_model

        algo = NormalPredictor()
        metrics = evaluate_model(algo, surprise_data, n_splits=2)
        for key in ["cv_rmse_mean", "cv_mae_mean", "cv_rmse_std", "cv_mae_std"]:
            assert key in metrics

    def test_train_final_model_rmse(self, surprise_data):
        from surprise import SVD
        from src.trainer import train_final_model

        algo = SVD(n_epochs=5, random_state=SEED)
        _, metrics = train_final_model(algo, surprise_data)
        assert "test_rmse" in metrics and "test_mae" in metrics
        assert 0 < metrics["test_rmse"] < 5

    def test_svd_rmse_better_than_baseline(self, surprise_data):
        from surprise import NormalPredictor, SVD
        from src.trainer import train_final_model

        _, baseline_m = train_final_model(NormalPredictor(), surprise_data)
        _, svd_m      = train_final_model(SVD(n_epochs=10, random_state=SEED), surprise_data)
        assert svd_m["test_rmse"] < baseline_m["test_rmse"]


# ── Tests recommender ─────────────────────────────────────────────────────────
class TestRecommender:

    def test_top_n_length(self, ratings_df, movies_df):
        from surprise import SVD
        from surprise.model_selection import train_test_split
        from src.recommender import get_top_n_recommendations
        from src.features import build_surprise_dataset

        data = build_surprise_dataset(ratings_df)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=SEED)
        algo = SVD(n_epochs=5, random_state=SEED)
        algo.fit(trainset)

        user_id  = ratings_df["userid"].iloc[0]
        all_ids  = ratings_df["movieid"].unique().tolist()
        seen_ids = set(ratings_df[ratings_df["userid"] == user_id]["movieid"])
        n = 5

        reco = get_top_n_recommendations(algo, user_id, all_ids, seen_ids, movies_df, n=n)
        assert len(reco) <= n
        assert "predicted_rating" in reco.columns
        assert (reco["predicted_rating"].between(*MODEL_PARAMS["SCALE"])).all()

    def test_no_already_seen_in_reco(self, ratings_df, movies_df):
        from surprise import SVD
        from surprise.model_selection import train_test_split
        from src.recommender import get_top_n_recommendations
        from src.features import build_surprise_dataset

        data = build_surprise_dataset(ratings_df)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=SEED)
        algo = SVD(n_epochs=5, random_state=SEED)
        algo.fit(trainset)

        user_id  = ratings_df["userid"].iloc[0]
        all_ids  = ratings_df["movieid"].unique().tolist()
        seen_ids = set(ratings_df[ratings_df["userid"] == user_id]["movieid"])

        reco = get_top_n_recommendations(algo, user_id, all_ids, seen_ids, movies_df, n=10)
        overlap = set(reco["movieid"]) & seen_ids
        assert len(overlap) == 0, f"Films déjà vus dans les reco : {overlap}"
