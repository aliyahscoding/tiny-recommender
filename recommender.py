# recommender.py
# Tiny item-item collaborative filter using cosine similarity.
# Usage examples:
#   python recommender.py --generate-data
#   python recommender.py --user U3 --top_n 5
#   python recommender.py --user U3 --top_n 5 --method sklearn

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ----------------------------- Data utilities ----------------------------- #

def generate_toy_ratings(csv_path: Path,
                         n_users: int = 10,
                         n_items: int = 10,
                         density: float = 0.6,
                         random_state: int = 7) -> pd.DataFrame:
    """
    Create a sparse long-form ratings table: columns[user_id, item_id, rating].
    density ~ fraction of user-item pairs that have a rating.
    """
    rng = np.random.default_rng(random_state)
    users = [f"U{i}" for i in range(1, n_users + 1)]
    items = [f"I{j}" for j in range(1, n_items + 1)]

    rows = []
    for u in users:
        for it in items:
            if rng.random() < density:
                rating = int(rng.integers(1, 6))  # 1..5
                rows.append((u, it, rating))

    # Ensure at least one rating per user and per item
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
    for u in users:
        if (df["user_id"] == u).sum() == 0:
            it = rng.choice(items)
            df.loc[len(df)] = (u, it, int(rng.integers(3, 6)))
    for it in items:
        if (df["item_id"] == it).sum() == 0:
            u = rng.choice(users)
            df.loc[len(df)] = (u, it, int(rng.integers(3, 6)))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def load_ratings(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[info] {csv_path} not found; generating toy data...")
        return generate_toy_ratings(csv_path)
    df = pd.read_csv(csv_path)
    expected_cols = {"user_id", "item_id", "rating"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"ratings.csv must have columns {expected_cols}")
    return df


def to_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to a user-item matrix; fill NaNs with 0 for 'unrated'."""
    mat = df.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
    mat = mat.reindex(index=sorted(mat.index), columns=sorted(mat.columns))
    return mat.fillna(0.0)


# --------------------------- Similarity functions ------------------------- #

def cosine_similarity_numpy(item_by_user: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between item vectors using NumPy.
    item_by_user shape: (n_items, n_users)
    Returns sim shape: (n_items, n_items)
    """
    # Normalize rows (items)
    norms = np.linalg.norm(item_by_user, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    X = item_by_user / norms
    sim = X @ X.T
    # zero self-similarity so it can't recommend the same item back
    np.fill_diagonal(sim, 0.0)
    return sim


def cosine_similarity_sklearn(item_by_user: np.ndarray) -> np.ndarray:
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise SystemExit(
            "[error] scikit-learn not installed. "
            "Install it with: pip install scikit-learn"
        )
    sim = cosine_similarity(item_by_user)
    np.fill_diagonal(sim, 0.0)
    return sim


# ---------------------------- Recommendation core ------------------------ #

def recommend_for_user(user_id: str,
                       user_item: pd.DataFrame,
                       item_sim: np.ndarray,
                       top_n: int = 5) -> pd.DataFrame:
    """
    Classic item-item scoring:
        score(item_j) = sum_i sim(j, i) * rating_u(i)
    Then normalize by the total similarity weight to avoid popularity bias.
    """
    if user_id not in user_item.index:
        raise KeyError(f"Unknown user_id '{user_id}'. Known: {list(user_item.index)}")

    # shape: users x items
    user_ratings = user_item.loc[user_id].to_numpy()            # (n_items,)
    rated_mask = user_ratings > 0                                # bool (n_items,)

    # sim is (n_items x n_items); multiply by user's ratings vector -> (n_items,)
    raw_scores = item_sim @ user_ratings

    # normalization by sum of absolute sim weights connected to user's rated items
    weight = (np.abs(item_sim) @ rated_mask.astype(float))       # (n_items,)
    scores = np.divide(raw_scores, np.maximum(weight, 1e-12))

    # Don't recommend items the user already rated
    scores[rated_mask] = -np.inf

    items = list(user_item.columns)
    top_idx = np.argsort(scores)[::-1][:top_n]
    recs = pd.DataFrame({
        "item_id": [items[i] for i in top_idx],
        "score": [float(scores[i]) for i in top_idx]
    })
    return recs


# ----------------------------------- CLI --------------------------------- #

def build(args) -> Tuple[pd.DataFrame, np.ndarray]:
    csv_path = Path(args.ratings)
    df = load_ratings(csv_path)
    user_item = to_user_item_matrix(df)  # users x items
    # transpose to items x users for similarity
    item_by_user = user_item.to_numpy().T

    if args.method == "sklearn":
        sim = cosine_similarity_sklearn(item_by_user)
    else:
        sim = cosine_similarity_numpy(item_by_user)
    return user_item, sim


def main():
    parser = argparse.ArgumentParser(description="Tiny item-item recommender (cosine similarity).")
    parser.add_argument("--ratings", default="data/ratings.csv", help="Path to ratings CSV")
    parser.add_argument("--generate-data", action="store_true", help="Generate toy ratings and exit")
    parser.add_argument("--user", default="U1", help="User ID to recommend for (e.g., U3)")
    parser.add_argument("--top_n", type=int, default=5, help="How many items to recommend")
    parser.add_argument("--method", choices=["numpy", "sklearn"], default="numpy",
                        help="Cosine similarity backend")
    args = parser.parse_args()

    if args.generate_data:
        df = generate_toy_ratings(Path(args.ratings))
        print(f"[ok] Generated {len(df)} ratings at {args.ratings}")
        return

    user_item, sim = build(args)
    recs = recommend_for_user(args.user, user_item, sim, top_n=args.top_n)

    print("\n=== Tiny Recommender ===")
    print(f"User: {args.user}")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
