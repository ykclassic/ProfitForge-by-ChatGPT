from sklearn.ensemble import RandomForestClassifier

class MetaModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200)

    def fit(self, X, y):
        self.model.fit(X, y)

    def probability(self, X):
        return self.model.predict_proba(X)[:, 1]
