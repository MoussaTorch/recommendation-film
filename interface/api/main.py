"""
FastAPI backend — CinéRec dashboard.

Démarrage :
    uvicorn interface.api.main:app --reload --port 8000
    (depuis le répertoire recommendation-film/)
"""
import sys
from pathlib import Path
from typing import Optional

import dill
import numpy as np
import pandas as pd
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from settings.params import RATINGS_INTERIM, MOVIES_INTERIM, MODEL_DIR, SEED
from src.data_loader import load_ratings, load_movies

app = FastAPI(title="CinéRec API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Métadonnées des modèles ───────────────────────────────────────────────────
MODEL_META: dict = {
    "svd": {
        "name": "SVD",
        "desc": "Factorisation matricielle (Simon Funk SVD)",
        "rmse": 0.853, "mae": 0.656,
        "is_best": True,
        "pattern": "*model_recommender.dill",
        "color": "#6366F1",
    },
    "svd_optuna": {
        "name": "SVD Optuna",
        "desc": "SVD optimisé via Optuna (TPE Sampler, 30 trials)",
        "rmse": 0.862, "mae": 0.663,
        "is_best": False,
        "pattern": "*svd_optuna_best.dill",
        "color": "#06B6D4",
    },
    "knn_user": {
        "name": "KNN User-Based",
        "desc": "K plus proches voisins utilisateurs (cosine, k=40)",
        "rmse": 0.940, "mae": 0.727,
        "is_best": False,
        "pattern": "*knn_user_based.dill",
        "color": "#F59E0B",
    },
    "knn_item": {
        "name": "KNN Item-Based",
        "desc": "K plus proches voisins items (cosine, k=40)",
        "rmse": 0.950, "mae": 0.743,
        "is_best": False,
        "pattern": "*knn_item_based.dill",
        "color": "#F97316",
    },
    "baseline": {
        "name": "Baseline",
        "desc": "NormalPredictor — distribution aléatoire (référence)",
        "rmse": 1.403, "mae": 1.120,
        "is_best": False,
        "pattern": None,
        "color": "#64748B",
    },
}

# ── État global ───────────────────────────────────────────────────────────────
_ratings: Optional[pd.DataFrame] = None
_movies: Optional[pd.DataFrame] = None
_models: dict = {}


def _load_all():
    global _ratings, _movies

    if not RATINGS_INTERIM.exists():
        raise FileNotFoundError(
            f"Données non trouvées : {RATINGS_INTERIM}. "
            "Lancez : python src/data_loader.py"
        )

    _ratings = load_ratings()
    _movies  = load_movies()
    logger.success(f"Données : {len(_ratings):,} notes | {_movies.shape[0]:,} films")

    # Charger les modèles depuis les fichiers dill
    for mid, meta in MODEL_META.items():
        pattern = meta.get("pattern")
        if pattern is None:
            _train_baseline()
            continue
        files = sorted(MODEL_DIR.glob(pattern))
        if files:
            with open(files[-1], "rb") as f:
                _models[mid] = dill.load(f)
            logger.success(f"[{mid}] {files[-1].name}")
        else:
            logger.warning(f"[{mid}] introuvable (pattern: {pattern})")


def _train_baseline():
    """Entraîne NormalPredictor à la volée (< 1 s)."""
    from surprise import NormalPredictor
    from src.features import build_surprise_dataset

    data     = build_surprise_dataset(_ratings)
    trainset = data.build_full_trainset()
    algo = NormalPredictor()
    algo.fit(trainset)
    _models["baseline"] = algo
    logger.success("[baseline] NormalPredictor entraîné")


@app.on_event("startup")
async def startup():
    try:
        _load_all()
    except FileNotFoundError as e:
        logger.warning(f"Démarrage sans données : {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "data_loaded": _ratings is not None,
        "models_loaded": list(_models.keys()),
    }


@app.get("/api/stats")
def stats():
    if _ratings is None:
        raise HTTPException(503, "Données non chargées — lancez data_loader.py")
    n = len(_ratings)
    u = int(_ratings["userid"].nunique())
    i = int(_ratings["movieid"].nunique())
    return {
        "n_ratings": n,
        "n_users": u,
        "n_items": i,
        "density_pct": round(n / (u * i) * 100, 2),
        "avg_rating": round(float(_ratings["rating"].mean()), 2),
        "min_rating": float(_ratings["rating"].min()),
        "max_rating": float(_ratings["rating"].max()),
        "models_available": list(_models.keys()),
    }


@app.get("/api/models")
def models_list():
    result = []
    for mid, meta in MODEL_META.items():
        result.append({
            "id": mid,
            "name": meta["name"],
            "desc": meta["desc"],
            "rmse": meta["rmse"],
            "mae": meta["mae"],
            "is_best": meta["is_best"],
            "available": mid in _models,
            "color": meta["color"],
        })
    return result


@app.get("/api/profile-methods")
def profile_methods():
    """Liste les méthodes de recommandation disponibles pour un profil libre."""
    available = list(_models.keys())
    methods = [
        {
            "id":        "knn_item",
            "name":      "KNN Similarité",
            "desc":      "Similarité cosine item-item (KNN Item-Based)",
            "badge":     "KNN",
            "color":     "#F97316",
            "available": "knn_item" in available,
        },
        {
            "id":        "svd_foldin",
            "name":      "SVD Fold-In",
            "desc":      "Estimation du vecteur utilisateur dans l'espace latent SVD",
            "badge":     "ML",
            "color":     "#6366F1",
            "available": "svd" in available,
        },
        {
            "id":        "genre_match",
            "name":      "Genre Match",
            "desc":      "Score d'overlap de genres pondéré par vos préférences",
            "badge":     "FAST",
            "color":     "#10B981",
            "available": _movies is not None,
        },
        {
            "id":        "popularity",
            "name":      "Popularité",
            "desc":      "Films les plus populaires (fallback universel)",
            "badge":     None,
            "color":     "#64748B",
            "available": _ratings is not None,
        },
    ]
    return methods


@app.get("/api/users")
def users(
    q:      str = Query(""),
    limit:  int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    if _ratings is None:
        raise HTTPException(503, "Données non chargées")

    stats_df = (
        _ratings
        .groupby("userid")
        .agg(n_ratings=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values("userid")
    )

    if q.strip():
        try:
            uid = int(q.strip())
            stats_df = stats_df[stats_df["userid"] == uid]
        except ValueError:
            stats_df = stats_df.head(0)

    total = len(stats_df)
    page  = stats_df.iloc[offset: offset + limit]

    result = []
    for _, row in page.iterrows():
        uid = int(row["userid"])
        top_genres: list[str] = []
        if _movies is not None:
            seen_ids = _ratings[_ratings["userid"] == uid]["movieid"]
            gl = _movies[_movies["movieid"].isin(seen_ids)]["genres_list"]
            if len(gl) > 0:
                from collections import Counter
                counts = Counter(
                    g for genres in gl if isinstance(genres, list) for g in genres
                )
                top_genres = [g for g, _ in counts.most_common(3)]

        result.append({
            "user_id":    uid,
            "n_ratings":  int(row["n_ratings"]),
            "avg_rating": round(float(row["avg_rating"]), 2),
            "top_genres": top_genres,
        })

    return {"total": total, "users": result}


class LikedMovie(BaseModel):
    movie_id: int
    rating:   float = 4.0


class ProfileRequest(BaseModel):
    liked_movies: list[LikedMovie] = []
    genres:       list[str]        = []
    method:       str               = "knn_item"   # knn_item | svd_foldin | genre_match | popularity
    n:            int               = 10


class RecommendRequest(BaseModel):
    user_id:  int
    model_id: str = "svd"
    n:        int = 10


@app.get("/api/genres")
def genres():
    if _movies is None:
        raise HTTPException(503, "Données non chargées")
    all_genres: set[str] = set()
    for gl in _movies["genres_list"].dropna():
        if isinstance(gl, list):
            all_genres.update(gl)
    return sorted(g for g in all_genres if g and g != "(no genres listed)")


@app.get("/api/movies/search")
def search_movies(q: str = Query(""), limit: int = Query(10, ge=1, le=30)):
    if _movies is None:
        raise HTTPException(503, "Données non chargées")
    if not q.strip():
        return []
    mask = _movies["title_clean"].str.contains(q.strip(), case=False, na=False)
    chunk = _movies[mask].head(limit)
    results = []
    for _, row in chunk.iterrows():
        gl_raw = row["genres_list"]
        gl = gl_raw.tolist() if hasattr(gl_raw, "tolist") else list(gl_raw) if gl_raw else []
        year_raw = row["year"]
        year = int(year_raw) if year_raw is not None and not (isinstance(year_raw, float) and pd.isna(year_raw)) else None
        results.append({
            "movie_id":   int(row["movieid"]),
            "title":      str(row["title_clean"]),
            "year":       year,
            "genres_list": gl,
        })
    return results


# ── Méthodes de recommandation profil ────────────────────────────────────────

def _scores_knn_item(liked_movies: list[LikedMovie], liked_ids: set[int]) -> dict[int, float]:
    """Similarité cosine item-item via le modèle KNN Item-Based."""
    if "knn_item" not in _models:
        raise HTTPException(503, "Modèle KNN Item-Based indisponible. Lancez trainer.py.")
    algo   = _models["knn_item"]
    scores: dict[int, float] = defaultdict(float)
    for item in liked_movies:
        weight = (float(item.rating) - 2.5) / 2.5
        if weight <= 0:
            continue
        try:
            inner_id = algo.trainset.to_inner_iid(int(item.movie_id))
        except ValueError:
            continue
        for nb_iid in algo.get_neighbors(inner_id, k=50):
            raw_id = int(algo.trainset.to_raw_iid(nb_iid))
            if raw_id not in liked_ids:
                scores[raw_id] += float(algo.sim[inner_id][nb_iid]) * weight
    return dict(scores)


def _scores_svd_foldin(liked_movies: list[LikedMovie], liked_ids: set[int]) -> dict[int, float]:
    """Estimation du vecteur utilisateur par fold-in dans l'espace latent SVD.
    p̂_u = Σ(w_i · q_i) / Σ|w_i|  puis  ŷ_uj = p̂_u · q_j + b_j + μ"""
    if "svd" not in _models:
        raise HTTPException(503, "Modèle SVD indisponible. Lancez trainer.py.")
    algo         = _models["svd"]
    weighted_sum = np.zeros(algo.qi.shape[1], dtype=np.float64)
    total_w      = 0.0
    for item in liked_movies:
        weight = (float(item.rating) - 2.5) / 2.5
        if weight <= 0:
            continue
        try:
            iid = algo.trainset.to_inner_iid(int(item.movie_id))
        except ValueError:
            continue
        weighted_sum += algo.qi[iid] * weight
        total_w      += abs(weight)
    if total_w == 0:
        return {}
    pu_est      = weighted_sum / total_w
    global_mean = algo.trainset.global_mean
    scores      = {}
    for inner_iid in range(algo.trainset.n_items):
        raw_id = int(algo.trainset.to_raw_iid(inner_iid))
        if raw_id not in liked_ids:
            score = float(np.dot(pu_est, algo.qi[inner_iid]) + algo.bi[inner_iid] + global_mean)
            scores[raw_id] = score
    return scores


def _scores_genre_match(liked_movies: list[LikedMovie], liked_ids: set[int]) -> dict[int, float]:
    """Score de correspondance par overlap de genres.
    Le profil de l'utilisateur est un vecteur de préférences par genre
    construit à partir de ses films notés."""
    genre_pref: dict[str, float] = defaultdict(float)
    for item in liked_movies:
        weight = (float(item.rating) - 2.5) / 2.5
        if weight <= 0:
            continue
        info = _movies[_movies["movieid"] == int(item.movie_id)]
        if len(info) == 0:
            continue
        gl = info["genres_list"].values[0]
        if isinstance(gl, list):
            for g in gl:
                genre_pref[g] += weight
    # Normaliser le vecteur de préférence
    total = sum(abs(v) for v in genre_pref.values()) or 1.0
    genre_pref = {g: v / total for g, v in genre_pref.items()}

    pop  = _ratings.groupby("movieid").size()
    pmax = float(pop.max()) if len(pop) else 1.0

    scores: dict[int, float] = {}
    for _, row in _movies.iterrows():
        mid = int(row["movieid"])
        if mid in liked_ids:
            continue
        gl = row["genres_list"]
        if not isinstance(gl, list):
            continue
        genre_score = sum(genre_pref.get(g, 0) for g in gl)
        if genre_score <= 0:
            continue
        pop_bonus = float(pop.get(mid, 0)) / pmax * 0.05
        scores[mid] = genre_score + pop_bonus
    return scores


def _scores_popularity(liked_ids: set[int]) -> dict[int, float]:
    """Films les plus populaires (nombre de notes), ignorant les films likés."""
    pop  = _ratings.groupby("movieid").size()
    pmax = float(pop.max()) if len(pop) else 1.0
    return {int(mid): float(cnt) / pmax
            for mid, cnt in pop.items() if int(mid) not in liked_ids}


def _build_results(scores: dict[int, float], genres: list[str], n: int, method_label: str) -> list[dict]:
    """Trie, filtre par genres, enrichit avec les métadonnées films."""
    if not scores:
        return []
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_score    = sorted_cands[0][1] if sorted_cands else 1.0
    results = []
    for movie_id, score in sorted_cands:
        if len(results) >= n:
            break
        info = _movies[_movies["movieid"] == movie_id]
        if len(info) == 0:
            continue
        gl_raw = info["genres_list"].values[0]
        gl = gl_raw.tolist() if hasattr(gl_raw, "tolist") else list(gl_raw) if gl_raw else []
        if genres and not any(g in gl for g in genres):
            continue
        year_raw = info["year"].values[0]
        year = int(year_raw) if year_raw is not None and not (isinstance(year_raw, float) and pd.isna(year_raw)) else None
        results.append({
            "rank":       len(results) + 1,
            "movie_id":   movie_id,
            "title":      str(info["title_clean"].values[0]),
            "genres_list": gl,
            "year":       year,
            "score":      round(score / max_score * 100, 1),
        })
    return results


@app.post("/api/recommend-profile")
def recommend_profile(req: ProfileRequest):
    """Recommandation profil libre (sans user_id dataset).
    Méthodes : knn_item | svd_foldin | genre_match | popularity"""
    if _ratings is None or _movies is None:
        raise HTTPException(503, "Données non chargées")

    liked_ids = {int(m.movie_id) for m in req.liked_movies}
    method    = req.method

    if method == "knn_item":
        scores = _scores_knn_item(req.liked_movies, liked_ids)
        if not scores:
            scores = _scores_popularity(liked_ids)
            method = "popularity_fallback"

    elif method == "svd_foldin":
        scores = _scores_svd_foldin(req.liked_movies, liked_ids)
        if not scores:
            scores = _scores_popularity(liked_ids)
            method = "popularity_fallback"

    elif method == "genre_match":
        scores = _scores_genre_match(req.liked_movies, liked_ids)
        if not scores:
            scores = _scores_popularity(liked_ids)
            method = "popularity_fallback"

    elif method == "popularity":
        scores = _scores_popularity(liked_ids)

    else:
        raise HTTPException(400, f"Méthode inconnue : '{method}'. Valides : knn_item, svd_foldin, genre_match, popularity")

    results = _build_results(scores, req.genres, req.n, method)

    return {
        "n_liked":       len(req.liked_movies),
        "genres_filter": req.genres,
        "method":        method,
        "recommendations": results,
    }


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    if _ratings is None:
        raise HTTPException(503, "Données non chargées")

    if req.model_id not in _models:
        available = list(_models.keys())
        raise HTTPException(
            404,
            f"Modèle '{req.model_id}' indisponible. "
            f"Disponibles : {available}. "
            "Lancez python src/trainer.py pour entraîner les modèles manquants."
        )

    if req.user_id not in _ratings["userid"].values:
        raise HTTPException(404, f"Utilisateur {req.user_id} introuvable dans le dataset.")

    algo     = _models[req.model_id]
    all_ids  = _ratings["movieid"].unique().tolist()
    seen_ids = set(int(m) for m in _ratings[_ratings["userid"] == req.user_id]["movieid"])
    unseen   = [m for m in all_ids if m not in seen_ids]

    if not unseen:
        raise HTTPException(400, f"L'utilisateur {req.user_id} a noté tous les films disponibles.")

    preds = [algo.predict(req.user_id, mid) for mid in unseen]
    preds.sort(key=lambda x: x.est, reverse=True)
    top = preds[: req.n]

    recommendations = []
    for i, pred in enumerate(top):
        iid  = pred.iid
        info = _movies[_movies["movieid"] == iid] if _movies is not None else pd.DataFrame()

        title       = str(info["title_clean"].values[0])  if len(info) else f"Film {iid}"
        genres_str  = str(info["genres"].values[0])        if len(info) else "N/A"
        genres_list_raw = info["genres_list"].values[0]    if len(info) else []
        year_raw        = info["year"].values[0]           if len(info) else None

        genres_list = (
            genres_list_raw.tolist()
            if hasattr(genres_list_raw, "tolist")
            else list(genres_list_raw) if genres_list_raw else []
        )
        year = int(year_raw) if year_raw is not None and not (isinstance(year_raw, float) and pd.isna(year_raw)) else None

        recommendations.append({
            "rank":             i + 1,
            "movie_id":         int(iid),
            "title":            title,
            "genres":           genres_str,
            "genres_list":      genres_list,
            "year":             year,
            "predicted_rating": round(float(pred.est), 3),
        })

    return {
        "user_id": req.user_id,
        "model":   req.model_id,
        "n_seen":  len(seen_ids),
        "n_unseen": len(unseen),
        "recommendations": recommendations,
    }
