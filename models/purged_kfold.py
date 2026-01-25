import numpy as np

class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.02):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, timestamps, label_end_times):
        n = len(X)
        fold_size = n // self.n_splits
        embargo = int(n * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size

            test_idx = np.arange(test_start, test_end)
            train_idx = []

            for j in range(n):
                if test_start - embargo <= j <= test_end + embargo:
                    continue
                if label_end_times[j] >= timestamps[test_start]:
                    continue
                train_idx.append(j)

            yield np.array(train_idx), test_idx
