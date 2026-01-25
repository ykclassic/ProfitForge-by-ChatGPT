from sklearn.metrics import accuracy_score, precision_score, recall_score

def walk_forward(model, df, features, target, train_size, test_size):
    results = []

    for i in range(train_size, len(df) - test_size, test_size):
        train = df.iloc[i - train_size:i]
        test = df.iloc[i:i + test_size]

        model.fit(train[features], train[target])
        preds = model.predict(test[features])

        results.append({
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "accuracy": accuracy_score(test[target], preds),
            "precision": precision_score(test[target], preds),
            "recall": recall_score(test[target], preds)
        })

    return results
