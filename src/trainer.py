"""
trainer.py
----------
Entraînement des modèles de recommandation avec tracking MLflow.
Modèles : Baseline (NormalPredictor), KNN User-Based, KNN Item-Based, SVD.
Optimisation : SVD + Optuna (run_svd_optuna).
"""
import sys
from pathlib import Path
from typing import Any

import dill
import mlflow
import numpy as np
import optuna
import pendulum
import pandas as pd
from loguru import logger
from optuna_integration import MLflowCallback
from surprise import Dataset, SVD, KNNBasic, NormalPredictor, accuracy
from surprise.model_selection import cross_validate, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings.params import (
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME,
    MODEL_DIR, MODEL_NAME, MODEL_PARAMS, SEED, TIMEZONE,
)

log_fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss!UTC}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"
)
logger.remove()
logger.add(sys.stderr, format=log_fmt, level="INFO")

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── MLflow ────────────────────────────────────────────────────────────────────
def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(
        f"MLflow — expérience : '{EXPERIMENT_NAME}' | URI : {MLFLOW_TRACKING_URI}"
    )


# ── Évaluation croisée ────────────────────────────────────────────────────────
def evaluate_model(algo, data: Dataset,
                   n_splits: int = MODEL_PARAMS["N_SPLITS"],
                   verbose: bool = False) -> dict[str, float]:
    """Cross-validation Surprise → RMSE et MAE moyens/std."""
    cv = cross_validate(algo, data, measures=["RMSE", "MAE"],
                        cv=n_splits, verbose=verbose)
    return {
        "cv_rmse_mean": round(float(np.mean(cv["test_rmse"])), 5),
        "cv_rmse_std" : round(float(np.std(cv["test_rmse"])), 5),
        "cv_mae_mean" : round(float(np.mean(cv["test_mae"])), 5),
        "cv_mae_std"  : round(float(np.std(cv["test_mae"])), 5),
    }


# ── Entraînement final ────────────────────────────────────────────────────────
def train_final_model(algo, data: Dataset) -> tuple[Any, dict]:
    """Split 80/20, entraîne sur train, évalue sur test."""
    trainset, testset = train_test_split(
        data, test_size=MODEL_PARAMS["TEST_SIZE"], random_state=SEED
    )
    algo.fit(trainset)
    preds = algo.test(testset)
    return algo, {
        "test_rmse": round(accuracy.rmse(preds, verbose=False), 5),
        "test_mae" : round(accuracy.mae(preds, verbose=False), 5),
    }


def save_model(model: Any, model_name: str) -> Path:
    """Sauvegarde le modèle avec dill (préfixe YYYYMMDD_)."""
    date_prefix = pendulum.now(tz=TIMEZONE).format("YYYYMMDD")
    path = MODEL_DIR / f"{date_prefix}_{model_name}"
    with open(path, "wb") as f:
        dill.dump(model, f)
    logger.success(f"Modèle sauvegardé : {path}")
    return path


# ── Runs individuels ──────────────────────────────────────────────────────────
def run_baseline(data: Dataset) -> dict:
    with mlflow.start_run(run_name="Baseline_NormalPredictor"):
        algo = NormalPredictor()
        cv_m = evaluate_model(algo, data)
        _, test_m = train_final_model(algo, data)
        mlflow.log_params({"algorithm": "NormalPredictor", "type": "baseline"})
        mlflow.log_metrics({**cv_m, **test_m})
        logger.info(f"[Baseline] test_rmse={test_m['test_rmse']}")
        return {**cv_m, **test_m}


def run_knn_user(data: Dataset) -> dict:
    p = MODEL_PARAMS["KNN_USER"]
    with mlflow.start_run(run_name="KNN_UserBased"):
        algo = KNNBasic(k=p["k"], sim_options=p["sim_options"])
        cv_m = evaluate_model(algo, data)
        trained, test_m = train_final_model(algo, data)
        mlflow.log_params({
            "algorithm" : "KNNBasic_UserBased",
            "k"         : p["k"],
            "similarity": p["sim_options"]["name"],
            "user_based": True,
        })
        mlflow.log_metrics({**cv_m, **test_m})
        save_model(trained, "knn_user_based.dill")
        logger.info(f"[KNN User] test_rmse={test_m['test_rmse']}")
        return {**cv_m, **test_m}


def run_knn_item(data: Dataset) -> dict:
    p = MODEL_PARAMS["KNN_ITEM"]
    with mlflow.start_run(run_name="KNN_ItemBased"):
        algo = KNNBasic(k=p["k"], sim_options=p["sim_options"])
        cv_m = evaluate_model(algo, data)
        trained, test_m = train_final_model(algo, data)
        mlflow.log_params({
            "algorithm" : "KNNBasic_ItemBased",
            "k"         : p["k"],
            "similarity": p["sim_options"]["name"],
            "user_based": False,
        })
        mlflow.log_metrics({**cv_m, **test_m})
        save_model(trained, "knn_item_based.dill")
        logger.info(f"[KNN Item] test_rmse={test_m['test_rmse']}")
        return {**cv_m, **test_m}


def run_svd(data: Dataset) -> dict:
    p = MODEL_PARAMS["SVD"]
    with mlflow.start_run(run_name="SVD"):
        algo = SVD(
            n_factors=p["n_factors"], n_epochs=p["n_epochs"],
            lr_all=p["lr_all"], reg_all=p["reg_all"],
            random_state=p["random_state"],
        )
        cv_m = evaluate_model(algo, data)
        trained, test_m = train_final_model(algo, data)
        mlflow.log_params({**p, "algorithm": "SVD"})
        mlflow.log_metrics({**cv_m, **test_m})
        model_path = save_model(trained, MODEL_NAME)
        mlflow.log_artifact(str(model_path))
        logger.info(f"[SVD] test_rmse={test_m['test_rmse']}")
        return {**cv_m, **test_m}


# ── Optimisation Optuna pour SVD ──────────────────────────────────────────────
def run_svd_optuna(data: Dataset) -> dict:
    """
    Optimise les hyperparamètres SVD avec Optuna + MLflowCallback.
    Chaque trial est loggé comme un run MLflow enfant.
    Le meilleur modèle est ré-entraîné et sauvegardé.
    """
    opt_cfg = MODEL_PARAMS["OPTUNA"]

    def objective(trial: optuna.Trial) -> float:
        n_factors = trial.suggest_int(
            "n_factors", *opt_cfg["n_factors_range"]
        )
        n_epochs = trial.suggest_int(
            "n_epochs", *opt_cfg["n_epochs_range"]
        )
        lr_all = trial.suggest_float(
            "lr_all", *opt_cfg["lr_range"], log=True
        )
        reg_all = trial.suggest_float(
            "reg_all", *opt_cfg["reg_range"], log=True
        )
        algo = SVD(
            n_factors=n_factors, n_epochs=n_epochs,
            lr_all=lr_all, reg_all=reg_all,
            random_state=SEED,
        )
        _, test_m = train_final_model(algo, data)
        return test_m["test_rmse"]

    mlflc = MLflowCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        metric_name="test_rmse",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    with mlflow.start_run(run_name="SVD_Optuna"):
        study = optuna.create_study(
            direction="minimize",
            study_name="svd_movie_recommendation",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(
            objective,
            n_trials=opt_cfg["n_trials"],
            show_progress_bar=True,
            callbacks=[mlflc],
        )

        best = study.best_params
        logger.info(f"[Optuna] meilleur RMSE : {study.best_value:.5f}")
        logger.info(f"[Optuna] meilleurs params : {best}")

        # Ré-entraîner le meilleur modèle sur tout le trainset
        best_algo = SVD(
            n_factors=best["n_factors"], n_epochs=best["n_epochs"],
            lr_all=best["lr_all"], reg_all=best["reg_all"],
            random_state=SEED,
        )
        _, test_m = train_final_model(best_algo, data)

        mlflow.log_params({**best, "algorithm": "SVD_Optuna_best"})
        mlflow.log_metrics({
            **test_m,
            "best_trial_rmse": study.best_value,
            "n_trials"       : len(study.trials),
        })

        model_path = save_model(best_algo, "svd_optuna_best.dill")
        mlflow.log_artifact(str(model_path))

        logger.info(f"[SVD Optuna] test_rmse final={test_m['test_rmse']}")
        return {**best, **test_m, "n_trials": len(study.trials)}


# ── Pipeline complet ──────────────────────────────────────────────────────────
def run_all_experiments(data: Dataset,
                        with_optuna: bool = False) -> pd.DataFrame:
    """Lance tous les modèles et retourne un tableau comparatif."""
    setup_mlflow()
    logger.info("=" * 60)
    logger.info("Démarrage des expériences MLflow")
    logger.info("=" * 60)

    results = {
        "Baseline" : run_baseline(data),
        "KNN_User" : run_knn_user(data),
        "KNN_Item" : run_knn_item(data),
        "SVD"      : run_svd(data),
    }
    if with_optuna:
        results["SVD_Optuna"] = run_svd_optuna(data)

    rows = [
        {
            "model"       : name,
            "cv_rmse_mean": r.get("cv_rmse_mean"),
            "cv_mae_mean" : r.get("cv_mae_mean"),
            "test_rmse"   : r.get("test_rmse"),
            "test_mae"    : r.get("test_mae"),
        }
        for name, r in results.items()
    ]
    df = pd.DataFrame(rows).sort_values("test_rmse")
    logger.info(f"\n{'='*60}\nRésultats :\n{df.to_string(index=False)}\n{'='*60}")
    return df


if __name__ == "__main__":
    from features import build_surprise_dataset
    from data_loader import load_ratings

    ratings = load_ratings()
    dataset = build_surprise_dataset(ratings)
    df_results = run_all_experiments(dataset, with_optuna=True)
