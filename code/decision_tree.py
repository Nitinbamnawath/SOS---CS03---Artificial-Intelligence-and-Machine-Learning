"""
Decision Tree Classifier
Week 3 - Summer of Science 2026

Dataset: Iris dataset (Fisher, 1936), loaded via sklearn.datasets.
Source/credit: R.A. Fisher, "The use of multiple measurements in taxonomic
problems", Annals of Eugenics, 1936. Distributed with scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------------------------------------------------
# 2. Train decision tree
# ---------------------------------------------------------
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc * 100:.1f}%")

# ---------------------------------------------------------
# 3. Plot the tree
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))
plot_tree(
    clf, feature_names=iris.feature_names,
    class_names=iris.target_names, filled=True, rounded=True, fontsize=9
)
plt.title("Decision Tree (max_depth=3) - Iris Dataset")
plt.tight_layout()
plt.savefig("/home/claude/sos_project/images/dtree_structure.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("/home/claude/sos_project/images/dtree_confusion.png", dpi=150)
plt.close()

print("Saved: dtree_structure.png, dtree_confusion.png")
