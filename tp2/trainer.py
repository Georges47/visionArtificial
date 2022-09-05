import numpy as np
from sklearn import tree
from joblib import dump
import matplotlib.pyplot as plt

hu_moments = np.loadtxt('./tp2/dataset.csv', delimiter=',', skiprows=1, usecols=range(0, 7))
labels = np.loadtxt('./tp2/dataset.csv', delimiter=',', skiprows=1, usecols=7)

# Training
classifier = tree.DecisionTreeClassifier().fit(hu_moments, labels)

# Show decision tree
tree.plot_tree(classifier)
plt.show()

# Save the decision tree model in a file
dump(classifier, './tp2/model.joblib')
