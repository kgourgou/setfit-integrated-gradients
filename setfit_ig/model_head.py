import torch


class SklearnToPyTorchLogisticRegression(torch.nn.Module):
    """
    Pass a trained sklearn LogisticRegression model to this class
    to create an equivalent PyTorch model.
    """

    def __init__(self, sklearn_model):
        super(SklearnToPyTorchLogisticRegression, self).__init__()

        # Extract the parameters from the sklearn model
        coef = sklearn_model.coef_.flatten()
        intercept = sklearn_model.intercept_.flatten()

        # Initialize the PyTorch parameters
        self.linear = torch.nn.Linear(coef.shape[0], 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(coef))
            self.linear.bias.copy_(torch.from_numpy(intercept))

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

    def predict(self, x):
        y_pred = self.forward(x).round().squeeze().int()
        return y_pred

    def predict_proba(self, x):
        # Compute the predicted probabilities of the positive class for input x
        return self.forward(x).squeeze()
