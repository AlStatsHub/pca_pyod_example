#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
np.random.seed(42) # replace 42 with any integer value
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles,make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from pyod.models.pca import PCA


name = [
    "PCA",
]

classifier = [
    PCA(), 
]


X, y = make_moons(n_samples=1500,noise=0.05, random_state=42)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_blobs(n_samples=1500, n_features=2, centers=2, random_state=42),
    make_circles(n_samples=1500,noise=0.01, factor=0.5, random_state=42),
    linearly_separable,
]

figure = plt.figure(figsize=(10,8))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.PuBu
    cm_bright = ListedColormap(["#FF8C00", "#483D8B"])

    ax = plt.subplot(len(datasets), len(classifier) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(name, classifier):
        ax = plt.subplot(len(datasets), len(classifier) + 1, i)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train)
        if isinstance(clf.named_steps['pca'], PCA):
            pca = clf.named_steps['pca']
            #print("Explained variance for", name, pca.explained_variance_)
            print("Explained variance ratio for", name, pca.explained_variance_ratio_)
        
        score = clf.decision_function(X_train)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score[0]).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()



# In[ ]:

output:
Explained variance ratio for PCA [0.96331924 0.03668076]
Explained variance ratio for P [0.5063987 0.4936013]
Explained variance ratio for P [0.634675 0.365325]

