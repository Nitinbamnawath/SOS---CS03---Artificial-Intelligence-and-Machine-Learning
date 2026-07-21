import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# single decision tree baseline
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, tree.predict(X_test))

# random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"single tree accuracy: {tree_acc*100:.1f}%")
print(f"random forest accuracy: {rf_acc*100:.1f}%")

# accuracy vs n_trees
n_range = [1, 2, 5, 10, 20, 50, 100, 200]
accs = []
for n in n_range:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    accs.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(6, 4))
plt.plot(n_range, accs, marker="o", color="seagreen")
plt.xlabel("Number of trees")
plt.ylabel("Test accuracy")
plt.title("Random Forest Accuracy vs. Number of Trees")
plt.tight_layout()
plt.savefig("../images/rf_ntrees.png", dpi=150)
plt.close()

# feature importances
importances = rf.feature_importances_
order = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(7, 5))
plt.barh([data.feature_names[i] for i in order][::-1], importances[order][::-1], color="steelblue")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("../images/rf_feature_importance.png", dpi=150)
plt.close()

# confusion matrix
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Greens", colorbar=False)
plt.title(f"Random Forest Confusion Matrix (Accuracy: {rf_acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("../images/rf_confusion.png", dpi=150)
plt.close()

print("top 5 features:", [data.feature_names[i] for i in order[:5]])
print("done")
