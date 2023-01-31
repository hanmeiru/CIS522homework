import numpy as np

class LinearRegression:

    """
    A linear regression model that uses closed-form formula to 
    calculate the weights and fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.random.normal(size=(1,))
        self.b = np.random.normal()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model with given independent dataset X and 
        dependent dataset y by updating the parameters w and b.

        Arguments:
            X (np.ndarray): the training input data.
            y (np.ndarray): the training output data. 

        Returns:
            None 

        """
        # X: (n,p), y:(n,)
        # w: (p,1), b: float
        one_vec = np.ones(len(X)).reshape(-1, 1)  # (n,1)
        X_augmented = np.concatenate((X, one_vec), axis=1)  # (n,p+1)
        if np.linalg.det(X_augmented.T @ X_augmented) != 0:
            all_para = np.linalg.inv(X_augmented.T @ X_augmented) @ (
                X_augmented.T @ y
            )  # (p+1,)
            self.w, self.b = all_para[:-1], all_para[-1]
        else:
            print("singular matrix...")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
          Predict the output for the given input.

          Arguments:
              X (np.ndarray): The input data.

          Returns:
              np.ndarray: The predicted output.

        """
        return X @ self.w + self.b  #


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 10, epochs: int = 1000000
    ) -> None:
        """
        Fit the linear regression model with given independent dataset X and 
        dependent dataset y by updating the parameters w and b.

        Arguments:
            X (np.ndarray): the training input data.
            y (np.ndarray): the training output data. 
            lr (float): the learning rate of training.
            epochs (int): number of training epochs.
            
        Returns:
            None 

        """
        n = X.shape[0]  # number of rows = 10320
        w_grad_total = 0  # for adaptive learning rate
        b_grad_total = 0
        for e in range(epochs):
            # predict y
            y_pred = self.predict(X)
            # calculate loss
            loss = (y_pred - y).T @ (y_pred - y) / n  # not used
            # calculate gradient:
            # for each observation i, partial_loss/partial w_j = 2(y_pred_i-y_i)x_ij/n
            # partial_loss / partial_b =  2(y_pred-y)/n
            w_grad = (y_pred - y) @ X * 2 / n  # (p,)
            b_grad = (y_pred - y).sum() * 2 / n  # scalar
            w_grad_total += abs(w_grad)
            b_grad_total += abs(b_grad)
            # update weights (using adaptive learning rate)
            self.w -= w_grad * (lr / (w_grad_total ** 0.5))
            self.b -= b_grad * (lr / (b_grad_total ** 0.5))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        if len(self.w) == X.shape[1]:
            return X @ self.w + self.b

        else:  # reinitialize w based on the size of X
            self.w = np.random.normal(size=(X.shape[1],))
            return X @ self.w + self.b

