import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Linear SVM (SVM)
clf1 = SVC(kernel='rbf')

# KNN (KNN)
clf2 = KNeighborsClassifier(n_neighbors=2000)

# Random Forest (RF)
clf3 = RandomForestClassifier()
