"""Shared code for The Engagement Trap notebooks.

Everything the notebooks reuse lives here: data loading, splitting, the four
recommenders, evaluation metrics, the diversity re-rankers, clustering, the
hidden "true user" oracle and the feedback-loop simulator.

Conventions:
- user and item ids are 0-based integers (re-mapped from MovieLens ids)
- a rating matrix R is a scipy CSR matrix of shape (n_users, n_items); a
  stored entry is a rating, an absent entry means "not rated"
- every model has fit(R), predict_all() -> predicted ratings (n_users, n_items)
  and rank_all() -> scores used to rank items for top-N lists
"""
import glob
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, silhouette_score

# set by load_movielens() once the data is loaded and filtered
N_USERS, N_ITEMS = None, None
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


# ---------------------------------------------------------------- data

def find_data_dir():
    """Folder containing ratings.csv / movies.csv (MovieLens ml-latest-small).
    Looks in Kaggle inputs first, then locally, and downloads from GroupLens
    as a last resort (needs internet)."""
    for pattern in ["/kaggle/input/**/ratings.csv", "../data/**/ratings.csv", "./data/**/ratings.csv"]:
        for hit in sorted(glob.glob(pattern, recursive=True)):
            folder = os.path.dirname(hit)
            if os.path.exists(os.path.join(folder, "movies.csv")):
                return folder
    os.makedirs("data", exist_ok=True)
    try:
        with urllib.request.urlopen(DATA_URL, timeout=30) as r, open("data/ml-latest-small.zip", "wb") as f:
            f.write(r.read())
    except OSError as e:
        raise FileNotFoundError(
            "ratings.csv/movies.csv not found under /kaggle/input and the download failed. "
            "Attach the MovieLens ml-latest-small dataset (Add Input) or turn Internet on.") from e
    zipfile.ZipFile("data/ml-latest-small.zip").extractall("data")
    return "data/ml-latest-small"


def load_raw(data_dir):
    """ratings.csv and movies.csv exactly as published (original ids)."""
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return ratings, movies


def load_movielens(data_dir=None, min_item_ratings=5):
    """Load, filter and re-index the data. Returns (ratings, items, genre_names).

    - movies with fewer than min_item_ratings ratings are dropped (their
      embeddings and similarities would rest on 1-4 ratings)
    - user and movie ids are re-mapped to 0..n-1 so they index matrix rows/cols
    - items get one 0/1 column per genre and a release year parsed from the title
    """
    global N_USERS, N_ITEMS
    raw_ratings, movies = load_raw(data_dir or find_data_dir())
    counts = raw_ratings["movieId"].value_counts()
    keep = counts[counts >= min_item_ratings].index
    raw_ratings = raw_ratings[raw_ratings["movieId"].isin(keep)]

    items = movies[movies["movieId"].isin(keep)].sort_values("movieId").reset_index(drop=True)
    items.insert(0, "item", np.arange(len(items)))
    user_ids = np.sort(raw_ratings["userId"].unique())
    ratings = pd.DataFrame({
        "user": np.searchsorted(user_ids, raw_ratings["userId"]),
        "item": raw_ratings["movieId"].map(dict(zip(items["movieId"], items["item"]))).to_numpy(),
        "rating": raw_ratings["rating"].to_numpy(),
        "ts": raw_ratings["timestamp"].to_numpy(),
    })

    genre_dummies = items["genres"].str.get_dummies(sep="|")
    genre_names = genre_dummies.columns.tolist()
    items = pd.concat([items, genre_dummies], axis=1)
    year = items["title"].str.strip().str.extract(r"\((\d{4})\)$")[0]
    items["year"] = pd.to_numeric(year, errors="coerce")

    N_USERS, N_ITEMS = len(user_ids), len(items)
    return ratings, items, genre_names


def genre_matrix(items, genre_names):
    """(n_items, n_genres) 0/1 matrix of genre flags."""
    return items[genre_names].to_numpy(dtype=float)


def item_features(items, genre_names):
    """Content features for the content-based model: genres + release decade one-hot."""
    year = items["year"].fillna(items["year"].mode()[0])
    decade = (year // 10 * 10).astype(int).astype(str) + "s"
    decade_onehot = pd.get_dummies(decade, prefix="decade").to_numpy(dtype=float)
    return np.hstack([genre_matrix(items, genre_names), decade_onehot])


def split_per_user(df, test_frac=0.2, seed=0):
    """Random per-user holdout: test_frac of each user's ratings go to test."""
    rng = np.random.default_rng(seed)
    key = pd.Series(rng.random(len(df)), index=df.index)
    rank = key.groupby(df["user"]).rank(method="first")
    n = df.groupby("user")["user"].transform("size")
    is_test = rank <= np.round(test_frac * n)
    return df[~is_test].copy(), df[is_test].copy()


def to_csr(df, shape=None):
    return sp.csr_matrix((df["rating"].to_numpy(dtype=float),
                          (df["user"].to_numpy(), df["item"].to_numpy())), shape=shape or (N_USERS, N_ITEMS))


def item_counts(df, n_items=None):
    """Number of ratings per item (= popularity)."""
    return np.bincount(df["item"], minlength=n_items or N_ITEMS).astype(float)


def center_rows(R):
    """Subtract each user's mean rating from their stored ratings.
    Returns (centered CSR, user means). Unrated cells stay absent (= 0)."""
    R = R.tocsr().astype(float)
    counts = np.diff(R.indptr)
    mu = np.asarray(R.sum(axis=1)).ravel() / np.maximum(counts, 1)
    Rc = R.copy()
    Rc.data -= np.repeat(mu, counts)
    return Rc, mu


def json_num(x, digits=5):
    """float for JSON; NaN becomes null (browsers reject NaN in JSON)."""
    return None if pd.isna(x) else round(float(x), digits)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, allow_nan=False)


def output_dir():
    """/kaggle/working on Kaggle, ./outputs locally; creates sub-folders."""
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
    for sub in ["figures", "results", "demo_data"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base


# ---------------------------------------------------------------- models

class Recommender:
    def rank_all(self):
        return self.predict_all()


class Popularity(Recommender):
    """Reference line: rank by number of ratings, predict a damped item mean."""

    def fit(self, R):
        R = R.tocsc()
        self.count = np.diff(R.indptr).astype(float)
        global_mean = R.sum() / R.nnz
        sums = np.asarray(R.sum(axis=0)).ravel()
        self.item_mean = (sums + 10 * global_mean) / (self.count + 10)
        self.n_users = R.shape[0]
        return self

    def predict_all(self):
        return np.tile(self.item_mean, (self.n_users, 1))

    def rank_all(self):
        return np.tile(self.count, (self.n_users, 1))


class BiasRidge(Recommender):
    """Exp 1 - Linear Regression. rating ~ b0 + b_user + b_item, fitted with
    Ridge on one-hot user and item columns."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, R):
        coo = R.tocoo()
        n, (n_users, n_items) = coo.nnz, R.shape
        rows = np.arange(n)
        X_user = sp.csr_matrix((np.ones(n), (rows, coo.row)), shape=(n, n_users))
        X_item = sp.csr_matrix((np.ones(n), (rows, coo.col)), shape=(n, n_items))
        X = sp.hstack([X_user, X_item]).tocsr()
        # sparse_cg supports sparse X together with fit_intercept=True
        model = Ridge(alpha=self.alpha, fit_intercept=True, solver="sparse_cg")
        model.fit(X, coo.data)
        self.b0 = model.intercept_
        self.b_user = model.coef_[:n_users]
        self.b_item = model.coef_[n_users:]
        return self

    def predict_all(self):
        return self.b0 + self.b_user[:, None] + self.b_item[None, :]


class ItemKNN(Recommender):
    """Exp 8 - KNN. Item-item cosine similarity on mean-centred ratings; a
    rating is predicted from the k most similar items the user has rated."""

    def __init__(self, k=40, shrink=25):
        self.k, self.shrink = k, shrink

    def fit(self, R):
        self.Rc, self.mu = center_rows(R)
        dot = (self.Rc.T @ self.Rc).toarray()
        norms = np.sqrt(np.asarray(self.Rc.power(2).sum(axis=0))).ravel() + 1e-9
        B = R.copy().tocsr()
        B.data[:] = 1.0
        co_raters = (B.T @ B).toarray()
        # shrink similarities that are based on only a few common users
        S = dot / np.outer(norms, norms) * co_raters / (co_raters + self.shrink)
        np.fill_diagonal(S, 0.0)
        self.S = np.clip(S, 0.0, None)  # keep positive neighbours only
        return self

    def predict_all(self):
        n_users, n_items = self.Rc.shape
        out = np.empty((n_users, n_items))
        for u in range(n_users):
            row = self.Rc[u]
            rated, dev = row.indices, row.data
            sims = self.S[:, rated]  # (n_items, n_rated)
            if len(rated) > self.k:  # zero out everything but the top-k per item
                kth = np.partition(sims, -self.k, axis=1)[:, [-self.k]]
                sims = np.where(sims >= kth, sims, 0.0)
            den = sims.sum(axis=1)
            num = sims @ dev
            out[u] = self.mu[u] + np.divide(num, den, out=np.zeros(n_items), where=den > 1e-9)
        return out


class SVDRec(Recommender):
    """Exp 6 - SVD. Truncated SVD of the user-mean-centred rating matrix
    (missing ratings = the user's mean): R_c ~ U S V^T."""

    def __init__(self, k=20, seed=0):
        self.k, self.seed = k, seed

    def fit(self, R):
        Rc, self.mu = center_rows(R)
        U, s, Vt = svds(Rc, k=self.k, random_state=self.seed)
        order = np.argsort(-s)  # svds returns singular values in ascending order
        self.U, self.s, self.Vt = U[:, order], s[order], Vt[order]
        return self

    def predict_all(self):
        return self.mu[:, None] + (self.U * self.s) @ self.Vt

    def item_vectors(self):
        """Item embeddings V*S: prediction deviation = U[u] . item_vectors[i]."""
        return self.Vt.T * self.s


class ContentPCA(Recommender):
    """Exp 5 - PCA. Item features (genres + decade) are compressed with PCA;
    each user gets their own ridge regression from those components to their
    centred ratings (the user's "taste profile")."""

    def __init__(self, features, n_components=8, alpha=1.0):
        self.features, self.n_components, self.alpha = features, n_components, alpha

    def fit(self, R):
        # PCA uses item metadata only (no ratings), so it cannot leak test data
        self.pca = PCA(n_components=self.n_components, random_state=0)
        self.Z = self.pca.fit_transform(self.features)  # (n_items, c)
        Rc, self.mu = center_rows(R)
        c = self.Z.shape[1]
        self.W = np.zeros((R.shape[0], c))
        for u in range(R.shape[0]):
            row = Rc[u]
            Zu = self.Z[row.indices]
            self.W[u] = np.linalg.solve(Zu.T @ Zu + self.alpha * np.eye(c), Zu.T @ row.data)
        return self

    def predict_all(self):
        return self.mu[:, None] + self.W @ self.Z.T


# ---------------------------------------------------------------- metrics

def top_n(scores, n):
    """Indices of the n highest scores per row, best first. (n_rows, n)"""
    idx = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def seen_mask(df, shape=None):
    mask = np.zeros(shape or (N_USERS, N_ITEMS), dtype=bool)
    mask[df["user"].to_numpy(), df["item"].to_numpy()] = True
    return mask


def recommend(scores, train_df, k=10):
    """Top-k lists that never repeat an item the user rated in train."""
    scores = scores.astype(float).copy()
    scores[seen_mask(train_df, scores.shape)] = -np.inf
    return top_n(scores, k)


def rating_metrics(pred_all, test_df):
    pred = pred_all[test_df["user"].to_numpy(), test_df["item"].to_numpy()]
    err = np.clip(pred, 0.5, 5) - test_df["rating"].to_numpy()
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err)))}


def ranking_metrics(top, test_df, n_items=None, threshold=4):
    """Precision/Recall/NDCG@k where relevant = test rating >= threshold.
    Users with no relevant test item are skipped."""
    n_users, k = top.shape
    n_items = n_items or N_ITEMS
    liked = test_df[test_df["rating"] >= threshold]
    rel = np.zeros((n_users, n_items), dtype=bool)
    rel[liked["user"].to_numpy(), liked["item"].to_numpy()] = True
    hits = np.take_along_axis(rel, top, axis=1).astype(float)  # (n_users, k)
    n_rel = rel.sum(axis=1)
    has_rel = n_rel > 0

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discount
    ideal = np.cumsum(discount)[np.clip(np.minimum(n_rel, k) - 1, 0, None)]
    return {
        f"Precision@{k}": float(hits.sum(1)[has_rel].mean() / k),
        f"Recall@{k}": float((hits.sum(1)[has_rel] / n_rel[has_rel]).mean()),
        f"NDCG@{k}": float((dcg[has_rel] / ideal[has_rel]).mean()),
    }


def gini(x):
    """0 = every item equally exposed, ~1 = all exposure on one item."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n, cum = len(x), np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def intra_list_diversity(top, G):
    """Mean pairwise (1 - cosine similarity) of genre vectors inside each list."""
    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
    V = Gn[top]  # (n_users, k, n_genres)
    sim = np.einsum("ukg,ujg->ukj", V, V)
    k = top.shape[1]
    off_diag = sim.sum(axis=(1, 2)) - np.einsum("ukk->u", sim)
    return float(np.mean(1 - off_diag / (k * (k - 1))))


def beyond_accuracy(top, G, popularity):
    exposure = np.bincount(top.ravel(), minlength=len(popularity))
    return {
        "Coverage": float((exposure > 0).mean()),
        "Gini": gini(exposure),
        "ILD": intra_list_diversity(top, G),
        "AvgPop": float(popularity[top].mean()),
    }


def evaluate(model, train_df, test_df, G, k=10):
    """All metrics for a fitted model."""
    pred = model.predict_all()
    # only the popularity reference ranks by something other than its predicted rating
    scores = model.rank_all() if isinstance(model, Popularity) else pred
    top = recommend(scores, train_df, k)
    return {**rating_metrics(pred, test_df),
            **ranking_metrics(top, test_df),
            **beyond_accuracy(top, G, item_counts(train_df))}


def evaluate_lists(top, train_df, test_df, G):
    """Metrics for already-built top-k lists (used for re-rankers)."""
    return {**ranking_metrics(top, test_df),
            **beyond_accuracy(top, G, item_counts(train_df))}


def mean_std_table(rows, group="model"):
    """rows: list of dicts (one per model per seed) -> 'mean ± std' table.
    Only numeric columns (other than seed) are summarised."""
    df = pd.DataFrame(rows)
    metrics = [c for c in df.select_dtypes("number").columns if c not in (group, "seed")]
    agg = df.groupby(group, sort=False)[metrics].agg(["mean", "std"])
    table = pd.DataFrame({m: agg[m]["mean"].map("{:.4f}".format) + " ± "
                          + agg[m]["std"].map("{:.4f}".format) for m in metrics})
    return table, agg


# ---------------------------------------------------------------- re-ranking

def normalize_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def candidate_pool(model_scores, seen, pool_size):
    """Top pool_size unseen items per user and their scores."""
    scores = model_scores.astype(float).copy()
    scores[seen] = -np.inf
    pool = top_n(scores, pool_size)
    return pool, np.take_along_axis(scores, pool, axis=1)


def mmr(pool, relevance, item_vecs_norm, lam, k=10):
    """Maximal Marginal Relevance. Greedily pick the item maximising
    lam * relevance - (1 - lam) * (max similarity to items already picked).
    lam = 1 -> plain ranking, lam = 0 -> maximum diversity."""
    rel = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-12)
    V = item_vecs_norm[pool]
    sim = V @ V.T
    picked = []
    max_sim = np.zeros(len(pool))
    available = np.ones(len(pool), dtype=bool)
    for _ in range(k):
        score = np.where(available, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(score))
        picked.append(j)
        available[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    return pool[picked]


def cluster_cap(pool, labels, cap, k=10):
    """Walk down the ranked pool, skipping items whose cluster already has
    `cap` items in the list. DBSCAN noise items (label -1) are never capped.
    If the pool runs out, fill the rest in plain rank order."""
    out, counts = [], {}
    for item in pool:
        c = labels[item]
        if c == -1 or counts.get(c, 0) < cap:
            out.append(item)
            counts[c] = counts.get(c, 0) + 1
            if len(out) == k:
                return np.array(out)
    chosen = set(out)
    rest = [item for item in pool if item not in chosen]
    return np.array(out + rest[:k - len(out)])


def rerank_all(pool, pool_scores, method, k=10, **kw):
    """Apply 'mmr' or 'cap' to every user's candidate pool. Returns (n_users, k)."""
    if method == "mmr":
        return np.array([mmr(pool[u], pool_scores[u], kw["item_vecs_norm"], kw["lam"], k)
                         for u in range(len(pool))])
    if method == "cap":
        return np.array([cluster_cap(pool[u], kw["labels"], kw["cap"], k)
                         for u in range(len(pool))])
    raise ValueError(method)


# ---------------------------------------------------------------- clustering (Exp 10)

def _expand(labels, active):
    """Labels for the active rows -> labels for all rows (-1 = not clustered)."""
    full = np.full(len(active), -1)
    full[active] = labels
    return full


def _silhouette(X, labels, seed=0):
    # sampled so the cost stays small for a few thousand movies
    return silhouette_score(X, labels, metric="cosine", sample_size=min(2000, len(X)), random_state=seed)


def choose_kmeans(X, ks=(5, 10, 20, 30, 40, 50), seed=0, active=None):
    """K-Means on L2-normalised embeddings; k picked by cosine silhouette.
    Rows where `active` is False (e.g. movies with no embedding yet) get -1."""
    active = np.ones(len(X), dtype=bool) if active is None else active
    Xn = normalize_rows(X[active])
    rows = []
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xn)
        rows.append({"k": k, "silhouette": _silhouette(Xn, labels, seed)})
    table = pd.DataFrame(rows)
    best_k = int(table.loc[table["silhouette"].idxmax(), "k"])
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(Xn)
    return _expand(labels, active), table


def choose_dbscan(X, eps_grid=None, min_samples=5, max_noise=0.5, active=None):
    """DBSCAN with cosine distance. eps picked by silhouette (on non-noise
    points) among settings with >= 2 clusters and at most max_noise noise.
    If none qualify, take the eps giving the most clusters.
    Rows where `active` is False get -1, like noise."""
    eps_grid = np.round(np.arange(0.05, 0.65, 0.05), 2) if eps_grid is None else eps_grid
    active = np.ones(len(X), dtype=bool) if active is None else active
    Xa = X[active]
    rows, all_labels = [], {}
    for eps in eps_grid:
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(Xa)
        core = labels != -1
        n_clusters = len(set(labels[core]))
        sil = np.nan
        if n_clusters >= 2 and core.sum() > n_clusters:
            sil = _silhouette(Xa[core], labels[core])
        rows.append({"eps": eps, "n_clusters": n_clusters,
                     "noise_frac": float((~core).mean()), "silhouette": sil})
        all_labels[eps] = labels
    table = pd.DataFrame(rows)
    ok = table[(table["n_clusters"] >= 2) & (table["noise_frac"] <= max_noise)].dropna()
    if len(ok):
        best_eps = ok.loc[ok["silhouette"].idxmax(), "eps"]
    else:
        best_eps = table.sort_values(["n_clusters", "noise_frac"],
                                     ascending=[False, True]).iloc[0]["eps"]
    return _expand(all_labels[best_eps], active), table, float(best_eps)


# ---------------------------------------------------------------- simulation

class LogisticOracle:
    """Exp 2 - Logistic Regression. The hidden 'true user' for the simulation:
    P(like = rating >= 4) from
      - user-genre affinity x item genres (one feature per genre)
      - user mean rating, damped item mean rating, log item popularity.
    It uses no SVD factors, so it does not share a model class with the
    recommender it is judging. Fit on a train split, AUC checked on held-out."""

    def fit(self, train_df, G, shrink=5):
        n_users, n_items = N_USERS, G.shape[0]
        self.G = G
        u, i = train_df["user"].to_numpy(), train_df["item"].to_numpy()
        mu = train_df.groupby("user")["rating"].mean().reindex(range(n_users)).to_numpy()
        dev = train_df["rating"].to_numpy() - mu[u]
        D = sp.csr_matrix((dev, (u, i)), shape=(n_users, n_items))
        B = sp.csr_matrix((np.ones(len(train_df)), (u, i)), shape=(n_users, n_items))
        # average centred rating the user gives to each genre (shrunk toward 0)
        self.affinity = np.asarray(D @ G) / (np.asarray(B @ G) + shrink)
        self.user_mean = mu
        count = item_counts(train_df, n_items)
        sums = np.bincount(train_df["item"], weights=train_df["rating"], minlength=n_items)
        global_mean = train_df["rating"].mean()
        self.item_mean = (sums + 10 * global_mean) / (count + 10)
        self.log_pop = np.log1p(count)

        X = self._features(u, i)
        self.lr = LogisticRegression(max_iter=2000).fit(X, train_df["rating"] >= 4)
        return self

    def _features(self, u, i):
        return np.hstack([self.affinity[u] * self.G[i], self.user_mean[u, None],
                          self.item_mean[i, None], self.log_pop[i, None]])

    def auc(self, df):
        X = self._features(df["user"].to_numpy(), df["item"].to_numpy())
        return float(roc_auc_score(df["rating"] >= 4, self.lr.predict_proba(X)[:, 1]))

    def prob_matrix(self):
        """P(like) for every (user, item). The model is linear in its features,
        so the full matrix is a few matrix products instead of 1.6M rows."""
        w, g = self.lr.coef_.ravel(), self.G.shape[1]
        logit = ((self.affinity * w[:g]) @ self.G.T
                 + w[g] * self.user_mean[:, None]
                 + w[g + 1] * self.item_mean[None, :]
                 + w[g + 2] * self.log_pop[None, :]
                 + self.lr.intercept_[0])
        return 1.0 / (1.0 + np.exp(-logit))


def seed_log(df, per_user=10, seed=0, shape=None):
    """Starting interaction history: per_user random real ratings per user."""
    sample = df.groupby("user", group_keys=False).sample(n=per_user, random_state=seed)
    return seen_mask(sample, shape)


def implicit_svd(log, k=20, seed=0):
    """SVD on the 0/1 click matrix. Returns (scores, item vectors)."""
    U, s, Vt = svds(sp.csr_matrix(log, dtype=float), k=k, random_state=seed)
    return (U * s) @ Vt, Vt.T * s


POLICIES = ["svd", "mmr", "kmeans_cap", "dbscan_cap"]


def simulate(policy, P_true, start_log, G, popularity, n_rounds=20, k=10, svd_k=20,
             pool_size=50, lam=0.7, cap=2, labels=None, seed=0, n_best=50):
    """Feedback loop. Each round the recommender is retrained on everything
    clicked so far, shows each user k items, and users click item i at
    position p with probability P_true[u, i] / log2(p + 2) (position bias).
    Clicks are added to the log. The recommender never sees P_true."""
    rng = np.random.default_rng(seed)
    log = start_log.copy()
    n_users, n_items = log.shape
    rows = np.arange(n_users)[:, None]
    position_bias = 1.0 / np.log2(np.arange(2, k + 2))
    exposure = np.zeros(n_items)
    shown = np.zeros_like(log)
    best = seen_mask(pd.DataFrame({"user": np.repeat(np.arange(n_users), n_best),
                                   "item": top_n(P_true, n_best).ravel()}), log.shape)
    history = []
    for r in range(n_rounds):
        scores, item_vecs = implicit_svd(log, svd_k, seed)
        pool, pool_scores = candidate_pool(scores, log, pool_size)
        if policy == "svd":
            recs = pool[:, :k]
        elif policy == "mmr":
            recs = rerank_all(pool, pool_scores, "mmr", k,
                              item_vecs_norm=normalize_rows(item_vecs), lam=lam)
        else:  # kmeans_cap / dbscan_cap
            recs = rerank_all(pool, pool_scores, "cap", k, labels=labels, cap=cap)

        clicks = rng.random(recs.shape) < P_true[rows, recs] * position_bias
        log[np.broadcast_to(rows, recs.shape)[clicks], recs[clicks]] = True
        exposure += np.bincount(recs.ravel(), minlength=n_items)
        shown[rows, recs] = True

        history.append({
            "round": r + 1,
            "CTR": float(clicks.mean()),                       # the proxy
            "TrueUtility": float(P_true[rows, recs].mean()),   # hidden P(like) of what was shown
            "Discovery": float((shown & best).sum() / best.sum()),  # share of each user's top-50 ever shown
            "CumCoverage": float((exposure > 0).mean()),
            "ExposureGini": gini(exposure),
            "ILD": intra_list_diversity(recs, G),
            "AvgPop": float(popularity[recs].mean()),
        })
    return pd.DataFrame(history)
