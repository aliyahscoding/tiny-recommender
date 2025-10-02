\# Tiny Item-Item Recommender



A minimal, portfolio-friendly item–item collaborative filtering demo in Python.  

It builds a user–item matrix from a tiny toy dataset, computes cosine similarity between items, and recommends top-N items for a chosen user.



\## How it works

1\. \*\*User–item matrix\*\*: pivot ratings into a matrix (users as rows, items as columns). Missing ratings are treated as 0.

2\. \*\*Item–item similarity\*\*: cosine similarity between item vectors  

&nbsp;  \\\[

&nbsp;  \\cos(\\theta) = \\frac{\\mathbf{x}\\cdot\\mathbf{y}}{\\|\\mathbf{x}\\|\\,\\|\\mathbf{y}\\|}

&nbsp;  \\]

3\. \*\*Scoring\*\*: for each unseen item j, score = sum over items i the user rated of `sim(j,i) \* rating(user,i)`, normalized by the total similarity weight.



\## Quickstart

```bash

python recommender.py --generate-data

python recommender.py --user U3 --top\_n 5





Use scikit-learn’s implementation if you prefer:

pip install scikit-learn

python recommender.py --user U3 --top\_n 5 --method sklearn





Files

recommender.py – all logic and a tiny CLI

data/ratings.csv – toy ratings (auto-generated)

requirements.txt – dependencies

.gitignore – keeps the repo clean







Notes

This is intentionally tiny and readable. Perfect for explaining item–item CF in interviews.

Extend by adding user-based CF, top-K neighbors, or a train/test split with metrics.

