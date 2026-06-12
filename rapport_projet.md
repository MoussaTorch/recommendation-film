# Rapport de Projet MLOps — Système de Recommandation de Films

**Auteur** : Moussa Seye — EPT DIC3
**Dataset** : MovieLens ml-latest-small — GroupLens Research
**Date** : Juin 2026

---

## Table des matières

1. [Introduction et objectif](#1-introduction-et-objectif)
2. [Dataset MovieLens](#2-dataset-movielens)
3. [Architecture MLOps du projet](#3-architecture-mlops-du-projet)
4. [Analyse exploratoire des données (EDA)](#4-analyse-exploratoire-des-données-eda)
5. [Pipeline de nettoyage et décisions](#5-pipeline-de-nettoyage-et-décisions)
6. [Feature Engineering — données processed](#6-feature-engineering--données-processed)
7. [Modèles de recommandation — théorie et implémentation](#7-modèles-de-recommandation--théorie-et-implémentation)
8. [Résultats et comparaison des modèles](#8-résultats-et-comparaison-des-modèles)
9. [Tracking MLflow](#9-tracking-mlflow)
10. [Tests automatisés](#10-tests-automatisés)
11. [Interface CinéRec — Dashboard et Prédiction](#11-interface-cinérec--dashboard-et-prédiction)
12. [Limitations et perspectives](#12-limitations-et-perspectives)
13. [Conclusion](#13-conclusion)
14. [Références](#14-références)

---

## 1. Introduction et objectif

Ce projet s'inscrit dans le module **MLOps** du cursus DIC3 de l'École Polytechnique de Thiès, sous la supervision de Mme Mously Diaw. L'objectif est double :

1. **Construire un système de recommandation de films fonctionnel** capable de prédire les préférences d'un utilisateur et de lui suggérer les films les plus susceptibles de lui plaire.
2. **Appliquer rigoureusement les bonnes pratiques MLOps** : traçabilité complète des expériences, versioning des modèles, pipeline reproductible, tests automatisés, et configuration centralisée.

### Pourquoi la recommandation de films ?

Les systèmes de recommandation sont omniprésents (Netflix, YouTube, Spotify, Amazon). Ils représentent un cas d'usage idéal pour MLOps car ils combinent des défis techniques concrets : données sparse, cold start, comparaison de plusieurs algorithmes, et besoin absolu de reproductibilité pour itérer de manière fiable.

Le jeu de données **MovieLens ml-latest-small** est le benchmark académique standard pour ce problème, fourni par le laboratoire GroupLens de l'Université du Minnesota, et utilisé dans des centaines de publications scientifiques.

---

## 2. Dataset MovieLens

### 2.1 Source et description générale

Le dataset est disponible sur le site de GroupLens Research. La version *ml-latest-small* est conçue pour la recherche et l'enseignement : assez grand pour être représentatif, assez petit pour s'entraîner localement.

Il est téléchargé automatiquement par `python src/data_loader.py` depuis :
`https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`

### 2.2 Structure des fichiers bruts

Le ZIP contient 4 fichiers CSV :


| Fichier       | Lignes  | Colonnes                           | Description                                               |
| ------------- | ------- | ---------------------------------- | --------------------------------------------------------- |
| `ratings.csv` | 100 836 | userId, movieId, rating, timestamp | Notes de 0,5 à 5 étoiles par paliers de 0,5               |
| `movies.csv`  | 9 742   | movieId, title, genres             | Titre avec année entre parenthèses, genres pipe-separated |
| `tags.csv`    | 3 683   | userId, movieId, tag, timestamp    | Tags libres déposés par les utilisateurs                  |
| `links.csv`   | 9 742   | movieId, imdbId, tmdbId            | Correspondance avec les bases IMDB et TMDB                |


**Ce qui est utilisé dans ce projet :** `ratings.csv` et `movies.csv`. Les tags ne sont pas utilisés dans la modélisation actuelle (filtrage collaboratif pur) mais sont archivés dans `data/interim/`. Les liens IMDB/TMDB ne sont pas utilisés.

### 2.3 Statistiques clés du dataset brut


| Indicateur                                | Valeur                         |
| ----------------------------------------- | ------------------------------ |
| Nombre de notes brutes                    | 100 836                        |
| Nombre d'utilisateurs                     | 610                            |
| Nombre de films                           | 9 724 (avec au moins une note) |
| Plage de notes                            | 0,5 à 5,0 (par paliers de 0,5) |
| Note moyenne                              | 3,50 / 5                       |
| Note médiane                              | 3,50 / 5                       |
| Films dans movies.csv (catalogue complet) | 9 742                          |
| Période couverte                          | 1996 à 2018                    |
| Densité matrice user-item                 | 1,70%                          |


### 2.4 Contexte académique du dataset

Le dataset MovieLens a été constitué à partir des notes réelles déposées sur le site MovieLens (movielens.org). Les utilisateurs ont été sélectionnés de façon aléatoire parmi ceux ayant noté au moins 20 films. Aucune information démographique n'est disponible. Les timestamps sont des secondes UNIX.

---

## 3. Architecture MLOps du projet

### 3.1 Structure du projet

```
recommendation-film/
├── data/
│   ├── raw/                      ← données brutes téléchargées (immuables)
│   │   └── ml-latest-small/      ← ratings.csv, movies.csv, tags.csv, links.csv
│   ├── interim/                  ← données nettoyées et filtrées
│   │   ├── ratings_filtered.csv  ← après filtre MIN_RATINGS=5
│   │   ├── movies_clean.csv      ← titre nettoyé, année extraite, genres_list
│   │   └── tags_clean.csv        ← tags standardisés
│   └── processed/                ← features finales prêtes pour la modélisation
│       ├── user_item_matrix.csv  ← matrice pivot 610 × 3 650
│       ├── user_features.csv     ← stats agrégées par utilisateur
│       ├── item_features.csv     ← stats par film + métadonnées
│       └── genre_matrix.csv      ← one-hot encoding des genres
├── interface/                    ← application web de démonstration
│   ├── api/
│   │   └── main.py               ← backend FastAPI (REST API, port 8000)
│   ├── src/
│   │   ├── views/
│   │   │   ├── Dashboard.jsx     ← statistiques + graphe RMSE + pipeline
│   │   │   └── Predict.jsx       ← 2 modes de prédiction en temps réel
│   │   └── components/           ← FilmCard, StatCard, Sidebar
│   ├── package.json              ← React 18 + Vite + Recharts
│   └── requirements.txt          ← FastAPI + Uvicorn
├── models/                       ← modèles sérialisés (.dill, gitignored)
├── notebooks/
│   └── 01-eda.ipynb              ← EDA complète avec MLflow logging
├── reports/                      ← graphiques générés par le notebook
├── src/
│   ├── data_loader.py            ← pipeline raw → interim
│   ├── features.py               ← pipeline interim → processed
│   ├── trainer.py                ← entraînement 5 modèles + Optuna + MLflow
│   └── recommender.py            ← inférence Top-N
├── settings/
│   └── params.py                 ← configuration centralisée (SEED, hyperparamètres, chemins)
├── tests/
│   └── test_pipeline.py          ← 18 tests unitaires pytest
├── requirements.txt
├── .gitignore
└── README.md
```

L'architecture suit le standard MLOps **raw → interim → processed** :

- **raw** : données brutes, jamais modifiées, immuables (source de vérité)
- **interim** : nettoyage et filtrage appliqués, justifiés par l'EDA
- **processed** : features finales construites à partir des données nettoyées

### 3.2 Pratiques MLOps appliquées


| Pratique                | Outil                | Ce que cela apporte                                                                                                     |
| ----------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Experiment tracking** | MLflow + SQLite      | Chaque entraînement est tracé : paramètres, métriques, artéfacts. Reproductibilité et comparaison objective entre runs. |
| **Versioning modèles**  | Dill + préfixe date  | `YYYYMMDD_nom.dill` dans `models/`. On sait exactement quel modèle a été entraîné quand.                                |
| **Logging structuré**   | Loguru               | Format UTC coloré avec niveau (INFO/SUCCESS/ERROR), fichier source et numéro de ligne. Facilite le débogage.            |
| **Config centralisée**  | `settings/params.py` | Un seul fichier pour SEED, hyperparamètres, chemins. Zéro chemin hardcodé dans le code.                                 |
| **Reproductibilité**    | SEED=42 partout      | Résultats identiques à chaque exécution (split, initialisation modèles).                                                |
| **Tests automatisés**   | pytest (18 tests)    | Garantit que le pipeline fonctionne après chaque modification.                                                          |
| **Gitignore strict**    | .gitignore           | Données brutes, modèles, mlruns, rapport exclus du dépôt.                                                               |


### 3.3 Paramètres centralisés (`settings/params.py`)

Tous les hyperparamètres du projet sont dans un seul fichier :

```python
SEED = 42          # reproductibilité totale
MODEL_PARAMS = {
    "TEST_SIZE"  : 0.20,        # 80% train / 20% test
    "N_SPLITS"   : 5,           # cross-validation 5-fold
    "TOP_N"      : 10,          # nombre de recommandations
    "MIN_RATINGS": 5,           # seuil cold start
    "SCALE"      : (0.5, 5.0),  # plage des notes MovieLens

    "SVD": { "n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02 },
    "KNN_USER": { "k": 40, "sim_options": {"name": "cosine", "user_based": True} },
    "KNN_ITEM": { "k": 40, "sim_options": {"name": "cosine", "user_based": False} },
    "OPTUNA": { "n_trials": 30, "n_factors_range": (50, 200), ... },
}
```

---

## 4. Analyse exploratoire des données (EDA)

L'EDA est conduite dans `notebooks/01-eda.ipynb`. Elle suit une logique structurée : **observer d'abord, décider ensuite**. Chaque graphe mène à une décision de nettoyage explicitement justifiée. Tous les graphiques sont sauvegardés dans `reports/` et loggés dans MLflow.

### 4.1 Qualité des données — valeurs manquantes et anomalies

**Graphe : `missing_values.png`**

Le graphe de gauche montre qu'il n'existe **aucune valeur NaN** dans `ratings.csv` (userId, movieId, rating, timestamp) ni dans `movies.csv` (movieId, title, genres). C'est exceptionnel pour un dataset réel et simplifie le nettoyage.

Le graphe de droite identifie les **anomalies non-NaN** :

- **0 doublon (userId, movieId)** : un utilisateur n'a noté un film qu'une seule fois → pas de dédoublonnage nécessaire.
- **34 films sans genre** : étiquetés `"(no genres listed)"` → marginaux (<0,4% du catalogue), conservés.
- **13 films sans année** : le regex `\(\d{4}\)` ne trouve pas d'année dans le titre → `year = NaN`, conservés.

**Conclusion** : le dataset est d'une qualité remarquable. Les seules anomalies sont marginales et ne nécessitent pas de suppression.

---

### 4.2 Distribution des notes

**Graphe : `distribution_notes.png`**

Le graphe de gauche (histogramme) montre la distribution des 100 836 notes :


| Note    | Nombre     | %         |
| ------- | ---------- | --------- |
| 0,5     | 1 370      | 1,4%      |
| 1,0     | 2 811      | 2,8%      |
| 1,5     | 1 791      | 1,8%      |
| 2,0     | 7 551      | 7,5%      |
| 2,5     | 5 550      | 5,5%      |
| **3,0** | **20 047** | **19,9%** |
| 3,5     | 13 136     | 13,0%     |
| **4,0** | **26 818** | **26,6%** |
| 4,5     | 8 551      | 8,5%      |
| **5,0** | **13 211** | **13,1%** |


**Interprétations clés :**

1. **La note 4,0 est la plus fréquente** (26 818 notes, 26,6%). Les utilisateurs notent positivement en majorité.
2. **82,3% des notes sont ≥ 3** : c'est le **biais de sélection positif** classique des systèmes de recommandation. Les gens regardent les films qu'ils pensent apprécier, et notent surtout ceux qu'ils ont regardés jusqu'au bout. Les films vraiment mauvais sont rarement notés.
3. **Les demi-notes sont moins fréquentes que les notes entières** : les utilisateurs préfèrent les valeurs rondes (phénomène psychologique documenté).

Le graphe de droite (KDE — densité) confirme une **distribution multimodale** avec des pics aux valeurs entières et demi-entières, et une concentration forte entre 3 et 5. La moyenne (3,50) et la médiane (3,50) sont confondues, ce qui signifie une distribution symétrique autour de ce point.

**Impact sur la modélisation** : cette asymétrie (peu de notes basses) peut biaiser les modèles vers des prédictions optimistes. Le SVD le gère grâce aux termes de biais (b_u, b_i) qui capturent les tendances individuelles.

---

### 4.3 Analyse du cold start — films trop peu notés

**Graphe : `cold_start_analysis.png`**

C'est le graphe le plus important de l'EDA car il justifie la principale décision de nettoyage.

**Graphe de gauche** : distribution du nombre de notes reçues par film (zoom sur 0–50, données brutes, 9 724 films) :

- La barre la plus haute (à gauche, avant le seuil rouge) regroupe **environ 4 600 films** avec seulement 1 note.
- La ligne rouge pointillée représente le **seuil MIN_RATINGS=5** choisi.
- L'annotation indique **6 074 films (62%) retirés** par ce filtre.

**Graphe de droite** : pourcentage du catalogue retiré selon le seuil choisi :

- Seuil 1 → 0% retiré (aucun filtre)
- Seuil 2 → 35% du catalogue retiré
- Seuil 3 → 49% retiré
- Seuil 4 → 57% retiré
- **Seuil 5 (notre choix, en rouge) → 62% retiré**
- Seuil 10 → 77% retiré
- Seuil 20 → 87% retiré

**Pourquoi choisir 5 ?**

Le Collaborative Filtering (KNN, SVD) prédit des notes en identifiant des *patterns de co-notation* : "les utilisateurs qui ont aimé ce film ont aussi aimé ces autres films". Un film noté 1 à 4 fois n'a **pas assez de voisins communs** pour que le modèle calcule une similarité fiable. La prédiction serait du bruit.

Le seuil 5 est le minimum standard dans la littérature (Harper & Konstan, 2015). Aller plus haut (10, 20) retire trop de films utiles. Aller plus bas (1, 2) laisse entrer du signal trop faible.

**Effet critique** : aucun utilisateur n'est retiré. Les 610 utilisateurs conservent toutes leurs notes. Ce sont uniquement les films "orphelins" qui disparaissent.

---

### 4.4 Distribution des genres

**Graphe : `genre_distribution_raw.png`** (catalogue brut, 9 742 films)

Le graphe horizontal montre la répartition des 20 genres dans le catalogue complet. Points clés :

- **Drama** est de loin le genre le plus représenté (~4 400 films). C'est une catégorie très large qui recouvre beaucoup de films.
- **Comedy** arrive en 2e position (~3 800 films). La combinaison Drama/Comedy représente environ 85% du catalogue.
- **Thriller et Action** (~1 900 chacun) forment le 2e tier.
- **Western, IMAX, Film-Noir** sont très minoritaires (<200 films) — risque de biais de popularité : ces genres seront moins bien couverts par le CF.
- `**(no genres listed)`** : 34 films, marginaux.

**Impact sur la modélisation** : le filtrage collaboratif n'utilise pas les genres directement. Cependant, le **biais de popularité** est réel : les genres rares (Western, Film-Noir) ont peu de notes, donc peu de signal pour les modèles KNN et SVD.

La **genre_matrix** (one-hot encoding des 20 genres) est construite pour des usages futurs (filtrage hybride, analyse Content-Based).

---

### 4.5 Évolution temporelle des notes

**Graphe : `temporal_analysis.png`**

Le graphe montre le volume de notes déposées par année, de 1996 à 2018 :

- **1996** : 6 000 notes → les premiers utilisateurs MovieLens, profil early adopter.
- **Pic en 2000** : ~10 000 notes — moment fort de l'adoption d'internet et de la culture cinéma en ligne.
- **Creux progressif 2008–2014** : entre 1 500 et 4 000 notes/an. Possible lassitude des utilisateurs ou mutation vers d'autres plateformes.
- **Regain 2015–2018** : entre 6 500 et 8 300 notes/an — la collecte de données s'est densifiée sur cette période.

**Ce que cela signifie pour la modélisation** : le dataset n'est **pas temporellement uniforme**. Un modèle qui utilise le timestamp (par exemple pour pondérer les notes récentes) pourrait améliorer les performances, mais ce n'est pas implémenté ici. Les modèles actuels (KNN, SVD) traitent toutes les notes de façon égale dans le temps.

---

### 4.6 Profil des utilisateurs (après nettoyage)

**Graphe : `user_features.png`**

**Graphe de gauche** — nombre de notes par utilisateur (données nettoyées, 610 users) :

- **La majorité des utilisateurs (>340) ont entre 12 et 100 notes** : c'est la masse centrale.
- **La médiane est à 68 notes** (ligne rouge pointillée) : la moitié des utilisateurs ont noté moins de 68 films.
- **Queue longue vers la droite** : quelques utilisateurs très actifs avec 500 à 2 200 notes. Ce sont des "super-utilisateurs" dont les préférences sont très bien capturées par le modèle.
- Cette distribution suit une **loi de puissance** (power law), typique des systèmes sociaux en ligne.

**Graphe de droite** — note moyenne par utilisateur :

- **Distribution quasi-normale centrée autour de 3,67**.
- La grande majorité des utilisateurs ont une note moyenne entre 3,0 et 4,5.
- Très peu d'utilisateurs sont systématiquement sévères (<2,5) ou systématiquement indulgents (>4,5).
- Cette homogénéité est favorable au modèle : les biais individuels (b_u dans SVD) sont limités.

---

### 4.7 Activité après nettoyage — users et films

**Graphe : `activity_users_items.png`** (échelle logarithmique, données nettoyées)

**Graphe de gauche** — notes par utilisateur (610 users, moyenne = 148 notes) :

- Axe X en log : les barres représentent des plages exponentielles.
- La première barre (10–20 notes) contient ~290 utilisateurs : la majorité des utilisateurs n'ont que très peu de notes même après le filtre.
- La moyenne (148 notes) est tirée vers le haut par les super-utilisateurs.

**Graphe de droite** — notes par film (3 650 films, moyenne = 25 notes) :

- Première barre (5–10 notes) : ~1 900 films — presque la moitié du catalogue filtré ne dépasse pas 10 notes.
- La médiane est à 13 notes (visible dans `item_distribution_clean.png`).
- Quelques films très populaires concentrent l'essentiel des notes (The Shawshank Redemption : 317 notes).

Ce graphe confirme la **double loi de puissance** : peu d'utilisateurs très actifs, beaucoup peu actifs ; peu de films très notés, beaucoup peu notés. C'est le pattern universel des systèmes de recommandation.

---

### 4.8 Sparsité de la matrice user-item

**Graphe : `sparsity_matrix.png`**

Ce graphe visualise un extrait de la matrice user-item (50 users × 100 films). Chaque point bleu foncé représente une note existante, le fond blanc représente l'absence de note.

**Sparsité globale : 95,95%** (après nettoyage, sur la matrice complète 610 × 3 650).

Ce que l'on voit clairement :

- **Quelques lignes très denses** (utilisateurs actifs qui ont noté beaucoup de films) — lignes quasi-continues de points bleus en haut.
- **La majorité des lignes très creuses** — quelques points isolés sur un fond blanc.
- **Aucun film n'est noté par tous les utilisateurs** — il n'y a aucune colonne entièrement remplie.

**C'est précisément pourquoi on utilise le Collaborative Filtering** plutôt que des approches directes : on ne peut pas simplement comparer les profils bruts, il faut inférer les préférences manquantes à partir des patterns de co-notation.

---

### 4.9 Note moyenne par genre

**Graphe : `genre_ratings.png`**

Ce graphe montre la note moyenne des films de chaque genre (données nettoyées). La ligne pointillée noire représente la moyenne globale (3,54).

**Genres au-dessus de la moyenne** (films mieux notés) :

- **Film-Noir** : ~3,90 — genre de niche très apprécié de ses amateurs. Peu de films, mais des classiques reconnus.
- **Documentary** : ~3,78 — les documentaires attirent un public engagé qui les note positivement.
- **War** : ~3,60 — les films de guerre de qualité (Apocalypse Now, Schindler's List) tirent la moyenne vers le haut.
- **Drama** : ~3,55 — légèrement au-dessus de la moyenne, cohérent avec sa place dominante.

**Genres en dessous de la moyenne** (films moins bien notés) :

- **Horror** : ~3,08 — genre le moins bien noté. Souvent perçu comme moins "artistique", et beaucoup de productions de faible qualité.
- **Children** : ~3,18 — films familiaux notés par des adultes qui les trouvent trop simples.
- **Comedy** : ~3,22 — la comédie est subjective et diversement appréciée.

**Ce que cela implique pour la recommandation** : un modèle qui ignore les genres peut recommander des films Horror à un utilisateur qui ne les apprécie pas. C'est l'une des limites du CF pur : il prédit bien *pour cet utilisateur* basé sur ses notes passées, mais n'exploite pas la connaissance des genres.

---

### 4.10 Corrélations entre features utilisateurs

**Graphe : `user_correlations.png`**

La heatmap montre les corrélations de Pearson entre les 5 features utilisateurs calculées :


| Paire                                  | Corrélation  | Interprétation                                                                                                                                      |
| -------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `user_n_ratings` ↔ `user_mean_rating`  | **-0,18**    | Les gros noteurs ont légèrement tendance à noter plus sévèrement. Les utilisateurs passionnés distinguent mieux le bon du mauvais.                  |
| `user_mean_rating` ↔ `user_min_rating` | **+0,50**    | Les utilisateurs généreux dans leur moyenne ont aussi des minimums plus élevés — cohérence interne des profils.                                     |
| `user_std_rating` ↔ `user_min_rating`  | **-0,63**    | Corrélation forte : les utilisateurs à forte variabilité (std élevé) mettent aussi de très basses notes. Plus on est discriminant, plus on met 0,5. |
| `user_n_ratings` ↔ `user_min_rating`   | **-0,31**    | Les super-utilisateurs ont tendance à noter certains films très bas — ils voient plus de films, dont davantage de mauvais.                          |
| `user_max_rating` ↔ autres             | proches de 0 | La note maximale (5) est presque universelle — presque tous les utilisateurs ont mis au moins un 5. Aucune information discriminante.               |


Ces corrélations sont toutes faibles à modérées, ce qui est **bon** : les features capturent des dimensions différentes et indépendantes du comportement utilisateur. Elles n'apportent pas d'information redondante.

---

### 4.11 Distribution des films après nettoyage

**Graphe : `item_distribution_clean.png`**

Ce graphe montre la distribution du nombre de notes par film dans les données nettoyées (3 650 films, zoom 0–100) :

- **Médiane : 13 notes** par film (ligne rouge).
- **La première barre (5–10 notes)** contient ~980 films — la majorité des films ont juste passé le filtre MIN_RATINGS=5.
- **Queue longue** : quelques dizaines de films très populaires avec 100+ notes (le pic à 100 sur le graphe représente tous les films avec ≥100 notes tronqués à 100).
- **Le film le plus noté** a 329 notes (Forrest Gump).

Ce comportement long tail est fondamental : **le modèle sera excellent pour les films populaires** (beaucoup de signal) et **moins précis pour les films de niche** (peu de signal). C'est une limite inhérente au CF.

---

## 5. Pipeline de nettoyage et décisions

Le pipeline `raw → interim` est implémenté dans `src/data_loader.py`. Chaque décision est justifiée par les observations de l'EDA.

### 5.1 Étape 1 — Filtrage des ratings (MIN_RATINGS = 5)

**Code** : `process_ratings()` dans `data_loader.py`

```
Avant : 100 836 notes | 610 users | 9 724 films
Après : 90 274 notes  | 610 users | 3 650 films
Conservé : 89,5% des notes
```

**Logique du filtre double** : on retire les utilisateurs ET les films ayant moins de 5 notes, puis on réitère jusqu'à convergence. En pratique, aucun utilisateur n'est retiré (tous avaient déjà ≥20 notes dans la sélection initiale de MovieLens). Seuls 6 074 films sont retirés.

**Effet sur la densité** : la matrice passe de **1,70% → 4,05%** (gain ×2,4). Un signal plus fort pour l'apprentissage.

**Résultat stocké** : `data/interim/ratings_filtered.csv` (2 076 KB)

### 5.2 Étape 2 — Nettoyage des films

**Code** : `process_movies()` dans `data_loader.py`

Trois transformations appliquées à `movies.csv` :

**a) Extraction de l'année** :
Le titre `"Toy Story (1995)"` contient l'année entre parenthèses. On l'extrait avec le regex `\(\d{4}\)$` pour créer une colonne `year`. Les 13 films sans format standard ont `year = NaN` et sont conservés.

**b) Nettoyage du titre** :
On retire l'année du titre pour créer `title_clean = "Toy Story"`. Cela améliore la lisibilité des recommandations et évite la redondance avec la colonne `year`.

**c) Conversion genres → liste Python** :
Le champ `genres = "Action|Comedy|Drama"` est converti en `genres_list = ["Action", "Comedy", "Drama"]`. Cela simplifie toutes les opérations analytiques (explode, get_dummies). Pour la sauvegarde CSV, on re-sérialise en pipe-separated pour éviter les conflits de parsing.

**Résultat** : `data/interim/movies_clean.csv` (894 KB) avec colonnes : movieid, title, genres, year, title_clean, genres_list.

### 5.3 Étape 3 — Archivage des tags

Les tags sont standardisés (colonnes en minuscules) et archivés dans `data/interim/tags_clean.csv` (112 KB). Non utilisés dans la modélisation actuelle, ils sont conservés pour des futures extensions (Content-Based, filtrage hybride).

---

## 6. Feature Engineering — données processed

Le pipeline `interim → processed` est implémenté dans `src/features.py`. Il construit 4 matrices de features sauvegardées dans `data/processed/`.

### 6.1 Matrice user-item (`user_item_matrix.csv`)

**Taille** : 610 lignes × 3 650 colonnes = 2 226 500 cellules

**Construction** : pivot de `ratings_filtered.csv` avec `userid` en index, `movieid` en colonnes, `rating` en valeurs. Les cellules non-renseignées sont `NaN`.

**Pourquoi** : c'est la représentation fondamentale du problème de recommandation. Chaque utilisateur est un vecteur de dimension 3 650 (très sparse). Les modèles de recommandation cherchent à compléter les valeurs manquantes de cette matrice.

**Utilisation** : analysée dans l'EDA pour calculer la sparsité et visualiser les patterns de notation. Sert de base pour l'encodage Surprise.

**Taille sur disque** : 2 460 KB (la plus grande des features processed, car 610 × 3 650 entiers).

### 6.2 Features utilisateurs (`user_features.csv`)

**Taille** : 610 lignes × 6 colonnes

**Colonnes calculées** (depuis `ratings_filtered.csv`) :


| Colonne            | Description                    | Utilité                                              |
| ------------------ | ------------------------------ | ---------------------------------------------------- |
| `userid`           | identifiant utilisateur        | clé de jointure                                      |
| `user_n_ratings`   | nombre total de notes déposées | mesure d'activité                                    |
| `user_mean_rating` | note moyenne de l'utilisateur  | biais positif/négatif personnel                      |
| `user_std_rating`  | écart-type des notes           | mesure de discrimination (le user est-il sélectif ?) |
| `user_min_rating`  | note minimale déposée          | profil du user le plus sévère                        |
| `user_max_rating`  | note maximale déposée          | presque toujours 5,0                                 |


**Pourquoi ces features** : elles caractérisent le *comportement de notation* de chaque utilisateur. Un utilisateur avec `user_mean_rating = 2,0` est beaucoup plus sévère qu'un autre avec `user_mean_rating = 4,5`. Ces features permettent de détecter les biais individuels, utiles pour des modèles hybrides ou des post-traitements.

**Taille sur disque** : 17 KB (leger, 610 lignes).

### 6.3 Features films (`item_features.csv`)

**Taille** : 3 650 lignes × 8 colonnes

**Construction** : jointure entre les statistiques des ratings et les métadonnées movies.

**Colonnes** :


| Colonne            | Source          | Description                      |
| ------------------ | --------------- | -------------------------------- |
| `movieid`          | ratings         | clé de jointure                  |
| `item_n_ratings`   | ratings (count) | popularité du film               |
| `item_mean_rating` | ratings (mean)  | qualité perçue moyenne           |
| `item_std_rating`  | ratings (std)   | consensus ou polarisation        |
| `title_clean`      | movies_clean    | titre sans année                 |
| `year`             | movies_clean    | année de sortie                  |
| `genres`           | movies_clean    | genres pipe-separated (original) |
| `genres_list`      | movies_clean    | genres en liste Python           |


**La jointure** (`LEFT JOIN sur movieid`) :
On joint les statistiques de notation (calculées sur `ratings_filtered.csv`) avec les métadonnées de `movies_clean.csv`. Cette jointure est un LEFT JOIN depuis les ratings : seuls les 3 650 films ayant passé le filtre MIN_RATINGS=5 sont conservés. Les 6 092 films de `movies_clean.csv` qui n'ont pas de notes sont exclus.

**Pourquoi** : cette table combine dans un seul endroit tout ce qu'on sait sur un film — sa popularité, sa qualité perçue, et ses métadonnées. C'est la table de référence pour générer les recommandations finales (titre affiché à l'utilisateur, genre pour l'analyse).

**Taille sur disque** : 302 KB.

### 6.4 Matrice genres one-hot (`genre_matrix.csv`)

**Taille** : 9 742 lignes × 22 colonnes

**Construction** : `pd.get_dummies()` appliqué sur `genres_list` après `explode()`. Chaque film est encodé par un vecteur binaire de 20 genres (+ movieid + title_clean = 22 colonnes).

**Pourquoi 9 742 lignes** (et non 3 650) : cette matrice couvre le **catalogue complet** de `movies_clean.csv`, pas seulement les films filtrés. Elle sert pour l'analyse Content-Based qui peut s'appliquer à n'importe quel film, y compris les plus rares.

**Exemple** :


| movieid | title_clean | Action | Animation | Comedy | Drama | ... |
| ------- | ----------- | ------ | --------- | ------ | ----- | --- |
| 1       | Toy Story   | 0      | 1         | 1      | 0     | ... |
| 2       | Jumanji     | 1      | 0         | 0      | 0     | ... |


**Utilisation actuelle** : analyse des notes par genre (graphe `genre_ratings.png`). **Utilisation future** : calcul de similarité Content-Based entre films (filtrage hybride).

**Taille sur disque** : 1 366 KB (plus grande que item_features car 9 742 lignes vs 3 650).

### 6.5 Dataset Surprise (en mémoire, non persisté)

La librairie scikit-surprise a son propre format de données. La fonction `build_surprise_dataset()` convertit le DataFrame ratings en objet `Dataset` via un `Reader(rating_scale=(0.5, 5.0))`. Ce format est nécessaire pour utiliser `cross_validate()`, `train_test_split()`, et tous les algorithmes de surprise.

Cet objet n'est pas sauvegardé sur disque — il est reconstruit à chaque exécution à partir des données interim (opération rapide, quelques secondes).

---

## 7. Modèles de recommandation — théorie et implémentation

Tous les modèles sont implémentés via la librairie **scikit-surprise** et tracés dans MLflow. Le module `src/trainer.py` orchestre l'entraînement.

### 7.1 Approche générale : Collaborative Filtering

Le **Collaborative Filtering (CF)** est la famille d'algorithmes la plus utilisée en recommandation. Son hypothèse fondamentale :

> *"Un utilisateur qui a eu des goûts similaires à d'autres utilisateurs dans le passé aura des goûts similaires dans le futur."*

Le CF n'utilise que l'historique de notes. Il ne regarde pas le contenu des films (genres, acteurs, synopsis). C'est sa force (generalise bien) et sa limite (ne peut pas expliquer "pourquoi" il recommande).

On distingue deux grandes familles :

- **Memory-Based** : KNN — calcule la similarité entre entités (users ou items) directement dans l'espace des notes.
- **Model-Based** : SVD — apprend un modèle paramétrique qui compresse l'information de la matrice.

---

### 7.2 Baseline — NormalPredictor

**Classe** : `surprise.NormalPredictor`

**Comment ça fonctionne** : ce modèle prédit une note aléatoire tirée d'une distribution normale ajustée sur les notes d'entraînement. Il estime μ (moyenne) et σ (écart-type) des notes, puis tire `ŷ ~ N(μ, σ²)` clampé dans [0,5 ; 5,0].

Il ne "prédit" rien d'utile : il ne tient compte ni de l'utilisateur ni du film. C'est un modèle **aléatoire informé uniquement par la distribution globale**.

**Rôle dans le projet** : c'est le plancher de performance. Si un modèle réel (KNN, SVD) ne fait pas mieux que ce baseline aléatoire, c'est qu'il ne fonctionne pas. Le test `test_svd_rmse_better_than_baseline` dans `test_pipeline.py` vérifie automatiquement cette propriété.

**Paramètres loggés dans MLflow** :

```json
{"algorithm": "NormalPredictor", "type": "baseline"}
```

**Résultats** : RMSE test = 1,403 | MAE test = 1,120. Une erreur moyenne de 1,4 étoiles est catastrophique — c'est le niveau "devinette éclairée".

---

### 7.3 KNN User-Based — Filtrage Collaboratif par Voisinage Utilisateurs

**Classe** : `surprise.KNNBasic(k=40, sim_options={"name": "cosine", "user_based": True})`

#### 7.3.1 Principe mathématique

L'idée : pour prédire la note de l'utilisateur *u* sur le film *i*, on cherche les *k* utilisateurs les plus similaires à *u* (ses "voisins") parmi ceux qui ont noté *i*, et on calcule une moyenne pondérée de leurs notes.

**Étape 1 — Calcul de la similarité cosinus entre utilisateurs**

Chaque utilisateur *u* est représenté par son vecteur de notes (sparse) :

```
r_u = [r_{u,1}, r_{u,2}, ..., r_{u,n}]  (NaN pour les films non notés)
```

La similarité cosinus entre deux utilisateurs *u* et *v* est calculée uniquement sur les **films qu'ils ont tous les deux notés** (ensemble I_{uv}) :

```
sim(u, v) = Σ_{i ∈ I_uv} r_{ui} × r_{vi}
            ─────────────────────────────────────────────────────
            √(Σ_{i ∈ I_uv} r_{ui}²) × √(Σ_{i ∈ I_uv} r_{vi}²)
```

La similarité cosinus vaut entre -1 (opposés) et 1 (identiques). Deux utilisateurs qui ont noté exactement les mêmes films avec les mêmes notes ont une similarité de 1.

**Étape 2 — Sélection des k voisins**

On retient les k=40 utilisateurs ayant la plus haute similarité avec *u* parmi ceux qui ont noté le film *i*.

**Étape 3 — Prédiction**

```
ŷ_{ui} = Σ_{v ∈ N_k(u,i)} sim(u,v) × r_{vi}
          ──────────────────────────────────────
          Σ_{v ∈ N_k(u,i)} |sim(u,v)|
```

C'est une moyenne pondérée par la similarité : les voisins plus similaires ont plus de poids.

#### 7.3.2 Avantages et limites

**Avantages** :

- Très intuitif : "les personnes qui me ressemblent aiment ce film"
- Interprétable : on peut expliquer "nous vous recommandons ce film parce que des utilisateurs comme vous l'ont adoré"
- Pas d'entraînement long : la similarité est calculée à la demande

**Limites** :

- **Scalabilité** : calculer les similarités entre N utilisateurs est en O(N²). Avec 610 utilisateurs c'est rapide, mais avec des millions c'est impraticable.
- **Cold start** : un nouvel utilisateur sans historique de notes ne peut pas être comparé à personne.
- **Sparsité** : si deux utilisateurs ont très peu de films en commun, la similarité calculée est peu fiable.

**Résultats** : RMSE test = 0,763 — meilleur que le baseline de 34%, mais moins bon que l'Item-Based.

---

### 7.4 KNN Item-Based — Filtrage Collaboratif par Voisinage Films

**Classe** : `surprise.KNNBasic(k=40, sim_options={"name": "cosine", "user_based": False})`

#### 7.4.1 Principe mathématique

Même logique que User-Based, mais on inverse la perspective : au lieu de comparer des utilisateurs, on compare des films.

**Calcul de la similarité entre films** :

Chaque film *i* est représenté par le vecteur des notes qu'il a reçues :

```
r_i = [r_{1,i}, r_{2,i}, ..., r_{m,i}]  (NaN pour les users n'ayant pas noté i)
```

La similarité cosinus entre films *i* et *j* est calculée sur les **utilisateurs qui ont noté les deux films** (ensemble U_{ij}).

**Prédiction** :

Pour prédire la note de *u* sur *i*, on cherche les k=40 films les plus similaires à *i* parmi ceux que *u* a déjà notés, et on fait une moyenne pondérée :

```
ŷ_{ui} = Σ_{j ∈ N_k(i,u)} sim(i,j) × r_{uj}
          ──────────────────────────────────────
          Σ_{j ∈ N_k(i,u)} |sim(i,j)|
```

#### 7.4.2 Pourquoi Item-Based > User-Based ?

Dans la littérature, l'Item-Based surpasse généralement l'User-Based. Dans nos résultats sur le vrai dataset (RMSE 0,940 User-Based vs 0,950 Item-Based), l'écart est faible et l'User-Based s'avère légèrement meilleur — phénomène explicable par le faible nombre d'utilisateurs (610) qui rend le voisinage utilisateurs très dense et fiable. Les deux raisons théoriques restent valides :

1. **Stabilité** : les films ne changent pas leurs préférences. Un film d'action reste similaire à d'autres films d'action. Les utilisateurs, en revanche, évoluent (nouvelle période de vie, goûts qui changent).
2. **Densité des voisinages** : avec 3 650 films et 90 274 notes, un film populaire a en moyenne 25 notes — suffisant pour calculer une similarité fiable entre films. Un utilisateur avec 68 notes (médiane) a moins d'overlap avec d'autres utilisateurs.
3. **Popularité des items** : il est plus facile de trouver des paires de films avec beaucoup d'utilisateurs en commun que des paires d'utilisateurs avec beaucoup de films en commun.

---

### 7.5 SVD — Décomposition en Valeurs Singulières (Matrix Factorization)

**Classe** : `surprise.SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)`

C'est le modèle principal et le plus performant du projet. C'est l'algorithme qui a dominé le **Netflix Prize** (2006–2009), la compétition qui a révolutionné les systèmes de recommandation.

#### 7.5.1 L'idée fondamentale : facteurs latents

Au lieu de calculer des similarités dans l'espace brut des notes (KNN), SVD **compresse l'information** de la matrice user-item dans un espace de dimension réduite appelé **espace des facteurs latents**.

L'intuition : si beaucoup d'utilisateurs qui aiment les films d'action "intenses" (Die Hard, Mad Max) aiment aussi les films d'arts martiaux (Kill Bill, Crouching Tiger), alors il existe un **facteur latent** qui capture cette préférence — appelons-le "action intense". Ce facteur n'est pas défini explicitement par nous ; il est **appris automatiquement** depuis les données.

#### 7.5.2 Le modèle mathématique

SVD dans scikit-surprise est en réalité l'algorithme de Simon Funk (2006), aussi appelé **biased Matrix Factorization**. La prédiction de la note de l'utilisateur *u* sur le film *i* est :

```
ŷ_{ui} = μ + b_u + b_i + q_i^T · p_u
```

Où :

- **μ** (mu) : biais global = note moyenne de toutes les notes (ici, ~3,54)
- **b_u** : biais utilisateur = tendance de l'utilisateur à noter plus haut ou plus bas que la moyenne. Si *u* est généreux, b_u > 0. Si *u* est sévère, b_u < 0.
- **b_i** : biais film = tendance du film à recevoir des notes plus hautes ou plus basses. Pour Shawshank Redemption (moyenne 4,43), b_i ≈ +0,89.
- **p_u** : vecteur de taille n_factors=100 représentant les **préférences latentes** de l'utilisateur (à quel point il aime chaque "concept" latent)
- **q_i** : vecteur de taille n_factors=100 représentant les **caractéristiques latentes** du film (à quel point le film incarne chaque "concept" latent)
- **q_i^T · p_u** : produit scalaire = affinité entre l'utilisateur et le film dans l'espace latent

#### 7.5.3 Apprentissage par descente de gradient stochastique (SGD)

Le modèle apprend les paramètres {b_u, b_i, p_u, q_i} en minimisant la **fonction de perte régularisée** sur les notes d'entraînement :

```
min   Σ_{(u,i,r) ∈ Train} (r_{ui} - ŷ_{ui})²
      + λ × (b_u² + b_i² + ||p_u||² + ||q_i||²)
```

Le premier terme est l'erreur quadratique (on veut prédire les notes avec précision). Le second terme est la **régularisation L2** qui évite l'overfitting (les paramètres ne doivent pas devenir trop grands).

**SGD — mise à jour des paramètres** à chaque note (u, i, r) :

```
e_{ui} = r_{ui} - ŷ_{ui}  (erreur de prédiction)

b_u  ← b_u  + lr × (e_{ui} - λ × b_u)
b_i  ← b_i  + lr × (e_{ui} - λ × b_i)
p_u  ← p_u  + lr × (e_{ui} × q_i  - λ × p_u)
q_i  ← q_i  + lr × (e_{ui} × p_u  - λ × q_i)
```

Avec `lr_all=0.005` (taux d'apprentissage) et `reg_all=0.02` (régularisation), sur `n_epochs=20` passes complètes sur les données, en utilisant `n_factors=100` facteurs latents.

#### 7.5.4 Hyperparamètres et leur rôle


| Hyperparamètre | Valeur | Rôle                                                                                                              |
| -------------- | ------ | ----------------------------------------------------------------------------------------------------------------- |
| `n_factors`    | 100    | Dimension de l'espace latent. Plus grand → plus expressif mais risque d'overfitting. 50-200 est standard.         |
| `n_epochs`     | 20     | Nombre de passes complètes sur les données d'entraînement. Plus d'époques → meilleure convergence mais plus long. |
| `lr_all`       | 0,005  | Taux d'apprentissage. Trop grand → instabilité. Trop petit → convergence lente.                                   |
| `reg_all`      | 0,02   | Pénalité de régularisation L2. Évite l'overfitting. Valeur standard dans la littérature.                          |
| `random_state` | 42     | Reproductibilité — initialisation des vecteurs latents.                                                           |


#### 7.5.5 Optimisation Optuna (`run_svd_optuna()`)

Pour trouver automatiquement les meilleurs hyperparamètres, `trainer.py` implémente une recherche avec **Optuna** (framework de bayesian optimization) :

- **30 trials** avec des valeurs de n_factors ∈ [50, 200], n_epochs ∈ [10, 50], lr_all ∈ [1e-4, 0.1] (log scale), reg_all ∈ [1e-4, 0.1] (log scale).
- Chaque trial est loggé comme un **run MLflow enfant** (runs imbriqués).
- L'algorithme d'exploration est **TPE (Tree-structured Parzen Estimator)** — une approche bayésienne qui apprend des essais précédents pour choisir intelligemment les prochaines valeurs à tester.
- Le meilleur modèle est ré-entraîné et sauvegardé sous `YYYYMMDD_svd_optuna_best.dill`.

#### 7.5.6 Pourquoi SVD est le meilleur modèle ?

1. **Compacité** : 100 facteurs capturent l'essentiel de la variance de la matrice (cf. théorème de Eckart-Young).
2. **Généralisation** : les facteurs latents extrapolent les préférences de l'utilisateur au-delà des films qu'il a notés. Un utilisateur n'a pas besoin d'avoir vu tous les films d'un genre pour que SVD détecte son affinité.
3. **Gestion de la sparsité** : SVD est conçu pour les matrices sparse — il n'utilise que les notes existantes pour l'entraînement.
4. **Biais intégrés** : les termes b_u et b_i absorbent les tendances systématiques, ce qui permet au produit q_i^T p_u de se concentrer sur les préférences pures.

---

## 8. Résultats et comparaison des modèles

### 8.1 Protocole d'évaluation

**Split** : 80% train / 20% test (`TEST_SIZE=0.20`, `SEED=42`)

**Cross-validation** : 5-fold pour estimer la variance des métriques (`N_SPLITS=5`)

**Métriques** :

- **RMSE (Root Mean Square Error)** : `√(Σ(r - ŷ)² / N)`. Pénalise fortement les grandes erreurs. Sur l'échelle [0,5 ; 5,0], une RMSE de 1,0 signifie une erreur moyenne d'environ 1 étoile.
- **MAE (Mean Absolute Error)** : `Σ|r - ŷ| / N`. Plus robuste aux outliers. Plus facile à interpréter : une MAE de 0,656 (SVD) signifie qu'on se trompe en moyenne de 0,656 étoiles.

### 8.2 Tableau des résultats

Résultats obtenus sur le **vrai dataset MovieLens ml-latest-small** (90 274 notes, 610 users, 3 650 films) :


| Modèle                     | RMSE test | MAE test  | CV RMSE moy. (5-fold) |
| -------------------------- | --------- | --------- | --------------------- |
| **SVD**                    | **0,853** | **0,656** | **0,857**             |
| KNN User-Based             | 0,940     | 0,727     | 0,945                 |
| KNN Item-Based             | 0,950     | 0,743     | 0,956                 |
| Baseline (NormalPredictor) | 1,403     | 1,120     | 1,409                 |


### 8.3 Analyse des résultats

**SVD est le meilleur modèle** avec un RMSE test de 0,853 :

- Il réduit l'erreur de **39% par rapport au Baseline** (1,403 → 0,853).
- Son MAE de 0,656 signifie qu'il se trompe en moyenne de **moins d'une étoile** sur l'échelle [0,5–5,0].
- Sa cohérence entre test (0,853) et CV (0,857) confirme qu'il ne sur-apprend pas.

**Ces résultats sont normaux et attendus pour ce dataset.** Dans la littérature, SVD sur MovieLens ml-latest-small donne typiquement entre 0,85 et 0,93 RMSE avec scikit-surprise. Le vrai dataset est beaucoup plus difficile que les données synthétiques (96% de sparsité, données réelles bruitées).

**KNN User-Based (RMSE 0,940) légèrement meilleur que KNN Item-Based (RMSE 0,950)** :
Résultat contre-intuitif par rapport à la littérature. Explication : avec seulement 610 utilisateurs et 90 274 notes (densité 4%), les profils utilisateurs sont relativement bien couverts — le voisinage user-based est dense et fiable. Les deux modèles restent proches et très largement au-dessus du baseline (−33% environ).

**Tous les modèles CF > Baseline** :
Ce résultat valide la pertinence de l'approche Collaborative Filtering. Le test automatisé `test_svd_rmse_better_than_baseline` garantit cette propriété à chaque exécution.

### 8.4 Exemple de recommandations Top-10 (utilisateur fictif)

Le module `recommender.py` génère les recommandations en :

1. Collectant tous les films non encore notés par l'utilisateur cible
2. Prédisant la note pour chaque film via le modèle SVD chargé
3. Triant par note prédite décroissante
4. Retournant les 10 premiers avec titre nettoyé, genres et note prédite

Exemple de sortie pour l'utilisateur 1 (films qu'il n'a pas vus) :

```
| movieid | title                       | genres            | predicted_rating |
|---------|------------------------------|-------------------|-----------------|
|     858 | Godfather, The               | Crime|Drama       |            4.72 |
|     527 | Schindler's List             | Drama|War         |            4.68 |
|    1221 | Godfather: Part II, The      | Crime|Drama       |            4.65 |
|     912 | Casablanca                   | Drama|Romance     |            4.61 |
|    ...  | ...                          | ...               |             ... |
```

La garantie `test_no_already_seen_in_reco` vérifie qu'aucun film déjà noté par l'utilisateur n'apparaît dans les recommandations.

---

## 9. Tracking MLflow

### 9.1 Configuration

```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "movie_recommendation"
```

Le backend SQLite est le choix moderne pour MLflow (le file store est déprécié en MLflow 3+). Toute la métadonnée (runs, params, metrics, artifacts paths) est stockée dans `mlflow.db`.

### 9.2 Ce qui est tracké par run

**Paramètres** (immuables, définis avant l'entraînement) :

- algorithme (NormalPredictor, KNNBasic_UserBased, KNNBasic_ItemBased, SVD)
- k, similarity, user_based (pour KNN)
- n_factors, n_epochs, lr_all, reg_all, random_state (pour SVD)

**Métriques** (calculées après entraînement) :

- cv_rmse_mean, cv_rmse_std (cross-validation)
- cv_mae_mean, cv_mae_std (cross-validation)
- test_rmse, test_mae (évaluation finale)

**Artéfacts** :

- Modèle sérialisé (`.dill`) pour SVD et KNN
- Graphiques EDA (`*.png`) loggés dans le run `EDA_full`
- Pour Optuna : chaque trial est un run MLflow enfant, permettant de visualiser l'évolution des hyperparamètres

### 9.3 Runs MLflow générés


| Run name                   | Modèle          | Type                           |
| -------------------------- | --------------- | ------------------------------ |
| `EDA_full`                 | —               | métriques dataset + graphiques |
| `Baseline_NormalPredictor` | NormalPredictor | baseline                       |
| `KNN_UserBased`            | KNNBasic (user) | modèle                         |
| `KNN_ItemBased`            | KNNBasic (item) | modèle                         |
| `SVD`                      | SVD standard    | modèle principal               |
| `SVD_Optuna`               | SVD optimisé    | parent + 30 runs enfants       |


### 9.4 Visualisation

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

L'interface permet de comparer tous les runs côte à côte, de filtrer par métrique, et de télécharger les artéfacts.

---

## 10. Tests automatisés

### 10.1 Organisation des 18 tests

```bash
pytest tests/ -v   # 18 passed ✅
```


| Classe                 | Tests   | Ce qui est vérifié                                                           |
| ---------------------- | ------- | ---------------------------------------------------------------------------- |
| `TestDataArchitecture` | 5 tests | Existence des répertoires raw/, interim/, processed/ et des fichiers interim |
| `TestDataLoader`       | 8 tests | Colonnes, échelle notes, doublons, filtre MIN_RATINGS, taille réelle dataset |
| `TestFeatures`         | 5 tests | Dimensions des matrices, types, encodage genres (valeurs 0/1 uniquement)     |
| `TestTrainer`          | 3 tests | Structure des métriques CV, RMSE dans [0,5], SVD > Baseline                  |
| `TestRecommender`      | 2 tests | Longueur Top-N ≤ N, aucun film déjà vu dans les reco                         |


### 10.2 Tests clés détaillés

`**test_real_dataset_size**` :

```python
assert len(ratings_df) > 80_000
```

Ce test garantit que le vrai dataset (100k notes) est présent et non les données synthétiques de test. Il échoue intentionnellement si les vraies données ne sont pas téléchargées.

`**test_svd_rmse_better_than_baseline**` :

```python
_, baseline_m = train_final_model(NormalPredictor(), surprise_data)
_, svd_m      = train_final_model(SVD(n_epochs=10, random_state=SEED), surprise_data)
assert svd_m["test_rmse"] < baseline_m["test_rmse"]
```

C'est le test le plus important du projet : il vérifie que le modèle principal apporte une vraie valeur ajoutée.

`**test_no_already_seen_in_reco**` :

```python
overlap = set(reco["movieid"]) & seen_ids
assert len(overlap) == 0
```

Garantit que le système ne recommande jamais un film que l'utilisateur a déjà noté.

---

## 11. Interface CinéRec — Dashboard et Prédiction

Pour rendre le projet démontrable et interactif, une interface web complète a été construite dans le sous-répertoire `interface/`. Elle constitue la couche applicative au-dessus de tout le pipeline ML.

### 11.1 Stack technique


| Couche         | Technologie                        | Rôle                                                  |
| -------------- | ---------------------------------- | ----------------------------------------------------- |
| **Backend**    | FastAPI + Uvicorn (Python)         | API REST, chargement des modèles `.dill`, prédictions |
| **Frontend**   | React 18 + Vite                    | Interface utilisateur dynamique, port 3000            |
| **Graphiques** | Recharts                           | BarChart horizontal comparant les RMSE                |
| **Style**      | CSS vanilla (design system custom) | Dark glassmorphism, tokens CSS                        |
| **Proxy**      | Vite dev server                    | Redirige `/api/`* → port 8000                         |


Démarrage :

```bash
# Backend
uvicorn interface.api.main:app --reload --port 8000

# Frontend
cd interface && npm run dev  # → http://localhost:3000
```

### 11.2 Dashboard — visualisation des résultats

Le dashboard (`/`) présente en temps réel (données chargées via l'API) :

- **4 cartes statistiques animées** :
  - Nombre total de notes (90 274)
  - Nombre d'utilisateurs (610)
  - Nombre de films (3 650)
  - Densité de la matrice (4,05%)
- **Graphe de comparaison des modèles** : BarChart horizontal (Recharts) avec le RMSE de chaque modèle. SVD est mis en avant (badge "BEST", barre indigo). Le graphe rend immédiatement visible la hiérarchie des performances.
- **Pipeline MLOps en 5 étapes** : visualisation séquentielle (Raw → Interim → Processed → Modèles → Interface) avec icône, titre et description pour chaque étape.

### 11.3 Module de prédiction — deux modes

Le module de prédiction (`/predict`) propose deux façons d'obtenir des recommandations réelles :

#### Mode 1 — Profil personnalisé (sans compte)

L'utilisateur construit son profil depuis zéro :

1. **Recherche autocomplete** : tape n'importe quel titre, l'API retourne les correspondances en 280 ms (debounce).
2. **Notation** : chaque film ajouté peut être noté de 1 à 5 étoiles via un widget interactif.
3. **Filtre par genres** : chips de sélection multiple (Drama, Thriller, Comedy…).
4. **Choix de la méthode de recommandation** — 4 algorithmes disponibles :


| Méthode            | Badge | Description                                                                           |
| ------------------ | ----- | ------------------------------------------------------------------------------------- |
| **KNN Similarité** | KNN   | Similarité cosine item-item — films les plus proches de vos films aimés               |
| **SVD Fold-In**    | ML    | Estime votre vecteur utilisateur dans l'espace latent SVD (p̂_u = Σwᵢ·qᵢ / Σ          |
| **Genre Match**    | FAST  | Construit un profil de préférences par genre à partir de vos notes, score par overlap |
| **Popularité**     | —     | Films les plus notés du catalogue (fallback universel)                                |


Chaque méthode utilise les vrais modèles entraînés. Le résultat est une prédiction ML réelle, pas un filtre statique.

#### Mode 2 — Utilisateur dataset (1 à 610)

L'utilisateur entre un ID dataset existant (ou tire au hasard) :

- Profil affiché : nombre de notes, note moyenne, genres favoris
- Choix du modèle parmi les 5 disponibles (SVD, SVD Optuna, KNN User, KNN Item, Baseline)
- Les prédictions sont les vraies sorties du modèle sur les films non encore vus par cet utilisateur

#### Résultats affichés

Chaque film recommandé est présenté dans une carte avec :

- Rang (#01, #02…)
- Titre et année (espacés proprement)
- Chips de genres colorées (chaque genre a sa couleur)
- Score de pertinence (jauge circulaire donut — arc cyan sur fond sombre, texte lisible)
- Note prédite (mode dataset) ou score normalisé (mode profil)

### 11.4 API REST — endpoints


| Méthode | Endpoint                 | Description                                     |
| ------- | ------------------------ | ----------------------------------------------- |
| GET     | `/api/health`            | Statut + modèles chargés                        |
| GET     | `/api/stats`             | Statistiques dataset (utilisé par le dashboard) |
| GET     | `/api/models`            | Liste des 5 modèles avec RMSE/MAE               |
| GET     | `/api/profile-methods`   | 4 méthodes de recommandation profil             |
| GET     | `/api/users?q=&limit=`   | Profils utilisateurs dataset avec pagination    |
| GET     | `/api/genres`            | Liste des genres disponibles                    |
| GET     | `/api/movies/search?q=`  | Recherche autocomplete (titre partiel)          |
| POST    | `/api/recommend`         | Prédictions pour un utilisateur dataset         |
| POST    | `/api/recommend-profile` | Prédictions pour un profil libre (4 méthodes)   |


---

## 12. Limitations et perspectives

### 11.1 Limitations actuelles

**Cold Start** : un utilisateur qui vient de s'inscrire sans historique de notes ne peut pas bénéficier du CF. Le modèle ne sait rien de lui. Solution : recommander les films les plus populaires, ou demander à l'utilisateur de noter quelques films de genres différents.

**Biais de popularité** : les films avec beaucoup de notes (blockbusters) sont mieux prédits que les films de niche. Le modèle tend à surestimer les films populaires. Solution : pondérer les similarités par la rareté (IDF-like weighting).

**CF pur, pas de contenu** : le modèle ignore les métadonnées des films (genres, réalisateur, acteurs). Deux films similaires en contenu seront bien liés si leurs audiences se recoupent, mais pas si leurs audiences sont distinctes.

**Pas de dimension temporelle** : les notes récentes ne sont pas pondérées plus que les anciennes. Un utilisateur dont les goûts ont évolué peut recevoir des recommandations obsolètes.

**Évaluation offline uniquement** : RMSE et MAE mesurent la précision des prédictions de notes, mais pas la satisfaction réelle de l'utilisateur. Une note prédite à 4/5 peut mener à un film que l'utilisateur aurait aimé à 3/5 — ce n'est pas la même chose qu'une mauvaise recommandation.

### 11.2 Perspectives d'amélioration

**Filtrage hybride (Content + Collaborative)** : combiner la matrice `genre_matrix.csv` (déjà construite) avec le CF. LightFM est une librairie conçue pour ça.

**Optimisation Optuna déjà implémentée** : lancer `python src/trainer.py` avec `with_optuna=True` pour découvrir automatiquement les meilleurs hyperparamètres SVD. Les 30 trials sont déjà configurés.

**ALS (Alternating Least Squares)** : alternative à SGD pour l'optimisation de la matrix factorization, souvent plus rapide sur les grands datasets.

**Déploiement API** : servir les recommandations en temps réel via une API FastAPI + Docker. Le modèle est déjà sérialisé (`.dill`) et chargeable en <1 seconde.

**Monitoring data drift** : surveiller l'évolution de la distribution des notes au fil du temps avec Evidently ou Alibi. Détecter quand le modèle doit être ré-entraîné.

---

## 13. Conclusion

Ce projet a permis de construire un **système de recommandation de films complet, reproductible et interactif**, en appliquant rigoureusement les bonnes pratiques MLOps sur chaque étape du pipeline — de l'ingestion des données brutes jusqu'à l'interface web de démonstration.

**Ce qui a été réalisé** :


| Composant           | Détail                                                                      |
| ------------------- | --------------------------------------------------------------------------- |
| Pipeline de données | 3 couches (raw → interim → processed), décisions justifiées par l'EDA       |
| Modèles entraînés   | 5 modèles : Baseline, KNN User, KNN Item, SVD, SVD Optuna                   |
| Meilleur modèle     | **SVD — RMSE 0,853** sur le vrai dataset MovieLens (−39% vs baseline 1,403) |
| Optimisation        | Optuna TPE, 30 trials automatiques, logs MLflow imbriqués                   |
| Tests               | **18 tests pytest** — tous passent, dont le test SVD > baseline             |
| Traçabilité         | MLflow + SQLite, 6 runs loggés, params + métriques + artéfacts              |
| Interface           | React 18 + FastAPI, dashboard temps réel, 4 méthodes de prédiction          |
| Reproductibilité    | SEED=42 partout, config centralisée, zéro chemin hardcodé                   |


**La leçon principale** : la qualité d'un système de recommandation ne tient pas seulement à l'algorithme choisi. Le nettoyage de données (filtrage MIN_RATINGS=5 qui multiplie la densité par 2,4x), la traçabilité (MLflow), et les tests automatisés sont tout aussi déterminants. Le projet démontre qu'un pipeline MLOps rigoureux produit des résultats fiables, comparables aux benchmarks publiés dans la littérature.

**L'interface CinéRec** concrétise le projet au-delà de la ligne de commande : elle permet à n'importe qui, sans compte ni historique dataset, de saisir des films qu'il a aimés et de recevoir en temps réel des recommandations générées par les vrais modèles entraînés — démontrant que les bonnes pratiques MLOps mènent à des systèmes déployables.

---

## 14. Références

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), article 19.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30–37.
- Funk, S. (2006). Netflix Update: Try This at Home. Blog post.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-Based Collaborative Filtering Recommendation Algorithms. *Proceedings of WWW 2001*.
- Hug, N. (2020). Surprise: A Python library for recommender systems. *Journal of Open Source Software*, 5(52), 2174.
- Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.
- MLflow Documentation — [https://mlflow.org/docs/latest/](https://mlflow.org/docs/latest/)
- Surprise Library — [https://surpriselib.com/](https://surpriselib.com/)
- GroupLens Research — [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

