# Summer of Science 2026 — AI and Machine Learning

**Mentee:** Nitin Bamnawath
**Mentor:** Sagnik Dey
**Institute:** IIT Bombay

This repo holds the code and results for my SoS project, organized by week. Each script is short and self-contained — run it directly to reproduce the plots in `images/`.

## Structure

```
code/
  linear_regression.py     # Week 2 - SLR with gradient descent from scratch
  logistic_regression.py   # Week 2 - binary classification, decision boundary
  kmeans.py                 # Week 3 - clustering + elbow method
  decision_tree.py          # Week 3 - classification tree on Iris

images/
  linreg_fit.png, linreg_cost.png
  logreg_boundary.png
  kmeans_clusters.png, kmeans_elbow.png
  dtree_structure.png, dtree_confusion.png
```

## Datasets used

- **Linear Regression:** synthetic data generated with NumPy (CGPA vs Package), styled after the example used in mentor's reference notes.
- **Logistic Regression:** synthetic 2D data via `sklearn.datasets.make_classification`.
- **K-Means:** synthetic blobs via `sklearn.datasets.make_blobs`.
- **Decision Tree:** Iris dataset (Fisher, 1936), loaded via `sklearn.datasets.load_iris`.

## Notes

These are from-scratch implementations of gradient descent for linear and logistic regression (no `sklearn.linear_model` used there), to make sure the underlying math is actually understood. K-Means and Decision Tree use scikit-learn directly since the goal there was understanding the algorithm behavior (elbow method, tree splits) rather than re-deriving the optimization.

Full write-up with explanations is in the midterm report PDF.
