import os
import numpy as np
from sklearn import tree
from joblib import dump
import matplotlib.pyplot as plt

hu_moments = np.loadtxt('tp2/csv/dataset.csv', delimiter=',', skiprows=0, usecols=(0, 1, 2, 3, 4, 5, 6))
debug_mode = False
labels = []
for index, element in enumerate(['square', 'star', 'triangle']):
    labels = labels + [index] * len(os.listdir(f'tp2/images/{element}'))

# Training
if debug_mode:
    print(hu_moments)
    print('Total Hu moments: ' + str(len(hu_moments)))
    print(labels)  # 0 = square, 1 = star, 2 = triangle, 3 = circle
    print('Squares: ' + str(len(list(filter(lambda x: x == 0, labels)))))
    print('Stars: ' + str(len(list(filter(lambda x: x == 1, labels)))))
    print('Triangles: ' + str(len(list(filter(lambda x: x == 2, labels)))))
    print('Total labels: ' + str(len(labels)))
classifier = tree.DecisionTreeClassifier().fit(hu_moments, labels)

# Show decision tree
tree.plot_tree(classifier)
if debug_mode:
    plt.show()

# Save the decision tree model in a file
dump(classifier, './tp2/model.joblib')
