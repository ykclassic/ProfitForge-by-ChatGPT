import numpy as np
from sklearn.linear_model import SGDClassifier

class OnlineModel:
    def __init__(self):
        self.model = SGDClassifier(loss="log_loss")
        self.classes = np.array([0, 1])
        self.initialized = False

    def update(self, X, y):
        if not self.initialized:
            self.model.partial_fit(X, y, classes=self.classes)
            self.initialized = True
        else:
            self.model.partial_fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
