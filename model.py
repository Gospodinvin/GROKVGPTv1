import numpy as np

class SimpleProbModel:
    """
    Лёгкая ML-модель без переобучения
    """

    def predict_proba(self, X):
        momentum = X[:, 0]
        volatility = X[:, 1]

        prob_up = 0.5 + 0.4 * momentum - 0.3 * volatility
        prob_up = np.clip(prob_up, 0.05, 0.95)

        return np.vstack([1 - prob_up, prob_up]).T
