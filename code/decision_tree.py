
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc * 100:.1f}%")

plt.figure(figsize=(12, 7))
plot_tree(
    clf, feature_names=iris.feature_names,
    class_names=iris.target_names, filled=True, rounded=True, fontsize=9
)
plt.title("Decision Tree (max_depth=3) - Iris Dataset")
plt.tight_layout()
plt.savefig("dtree_structure.png")
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("dtree_confusion_.png")
plt.show()

