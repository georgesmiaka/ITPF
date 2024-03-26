from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score

class BlackBox:
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, X):
        return

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    @abstractmethod
    def predict_proba(self, X):
        return

class BlackBoxWrapper(BlackBox):
    def __init__(self, model):
        self.clf = model
    def predict(self, X):
        # here the input is 2-d and we want to transform it to 3-d before prediction
        x = X.reshape((X.shape[0], X.shape[1], 1))
        y = self.clf.predict(x)
        return y
    