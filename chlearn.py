import numpy as np

class KNeighborsRegressor:
    def __init__(self, n_neighbors: int = 5):
        """
        Initialize the K-Nearest Neighbors regressor.
        :param n_neighbors: Number of neighbors to consider for prediction
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Store the training data for later use in prediction.
        :param X: Independent variable (2D numpy array)
        :param y: Dependent variable (1D numpy array)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for input X using the K-Nearest Neighbors algorithm.
        :param X: New independent variable (2D numpy array)
        :return: Predicted values (1D numpy array)
        """
        X = np.array(X)
        predictions = []

        for x in X:
            # Calculate Euclidean distances from x to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of the k-nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]

            # Compute the mean of the target values of the k-nearest neighbors
            predicted_value = np.mean(self.y_train[nearest_indices])
            predictions.append(predicted_value)

        return np.array(predictions)



class LinearRegression:
    def __init__(self):
        self.slope = None  # weight
        self.intercept = None  # bias


    def fit(self, X, y):
        """
        learning function
        :param X: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        denominator = np.sum(pow(X-X_mean, 2))
        numerator = np.sum((X-X_mean)*(y-y_mean))

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * X_mean)


    def predict(self, X) -> np.ndarray:
        """
        predict value for input
        :param X: new indepent variable
        :return: predict value for input (2d array format)
        """
        return self.slope * np.array(X) + self.intercept