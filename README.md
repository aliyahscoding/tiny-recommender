# Tiny Item-Item Recommender

A minimal, portfolio-friendly item–item collaborative filtering demo in Python. It builds a user–item matrix from a toy dataset, computes cosine similarity between items, and recommends top-N items for a chosen user.

## How it works

1. **User–item matrix**: pivot ratings into a matrix (users as rows, items as columns). Missing ratings are treated as 0.  
2. **Item–item similarity**: compute cosine similarity between item vectors:

   $$
   \cos(\theta) = \frac{\mathbf{x}\cdot\mathbf{y}}{\lVert \mathbf{x}\rVert\,\lVert \mathbf{y}\rVert}
   $$

3. **Scoring**: for each unseen item \(j\),
   \[
   \text{score}(j)=\sum_i \text{sim}(j,i)\times \text{rating}(u,i)
   \]
   then normalize by the total similarity weight to avoid popularity bias.

## Quickstart
```bash
# 1) Create toy ratings (writes to data/ratings.csv)
python recommender.py --generate-data

# 2) Get top-5 recommendations for user U3
python recommender.py --user U3 --top_n 5
```

## Use scikit-learn's implementation if you prefer:
```bash
pip install scikit-learn
python recommender.py --user U3 --top_n 5 --method sklearn
```

## Example Output
```bash
=== Tiny Recommender ===
User: U3
 item_id  score
 I7       0.8123
 I2       0.7035
 I9       0.6461
 I4       0.5902
 I1       0.5718
```
(Number's will differ because toy data is different)

## Files
**recommender.py** — all logic and a tiny CLI
**data/ratings.csv** — toy ratings (auto-generated)
**requirements.txt** — dependencies
**.gitignore** — keeps the repo clean

## Data format
Long-form CSV with three columns:
user_id	item_id	rating
U1	I3	4
U1	I7	5
U2	I1	2


## Notes
Intentionally tiny and readable. Good for explaining item–item CF in interviews.
Easy extensions: user-based CF, top-K neighbors, or a simple train/test split with evaluation metrics.


