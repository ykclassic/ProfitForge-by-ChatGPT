from river.drift import ADWIN

class DriftDetector:
    def __init__(self):
        self.adwin = ADWIN()

    def update(self, error):
        self.adwin.update(error)
        return self.adwin.drift_detected
