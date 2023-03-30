import torch
import torch.nn as nn
from tqdm import tqdm


class BinaryLogisticRegressionModel(nn.Module):
    """
    Implementation of Binary Logistic Regression.
    """

    def __init__(
        self, input_dimension, lr=0.01, number_of_epochs=100, device="cpu"
    ):
        super(BinaryLogisticRegressionModel, self).__init__()

        self.input_dimension = input_dimension
        self.linear = nn.Linear(input_dimension, 1, device=device)
        self.lr = lr
        self.number_of_epochs = number_of_epochs
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=1.0
        )
        self.device = device

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y):
        if isinstance(y, list):
            y = torch.tensor(y, dtype=torch.float, device=self.device)

        for epoch in tqdm(range(self.number_of_epochs)):
            self.optimizer.zero_grad()
            y_prediction = self(X).flatten()
            loss = self.criterion(y_prediction, y)
            loss.backward()
            self.optimizer.step()

            if epoch % 10000 == 0:
                print(f"logreg loss = {loss.item()}")

    def predict(self, X, thr=0.5):
        # predict classes
        return torch.sigmoid(self(X)) > thr

    def predict_proba(self, X) -> float:
        """
        Returns only the P(y=1|X).
        """
        return torch.sigmoid(self(X)).flatten()
