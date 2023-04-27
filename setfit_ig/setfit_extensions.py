import copy
from collections import OrderedDict, namedtuple

import torch
from sentence_transformers.SentenceTransformer import SentenceTransformer
from setfit import SetFitModel
from torch.autograd import grad
from torch.nn import Sequential


class SetFitModelWithTorchHead(SetFitModel):
    """
    This follows SetFitModel, except that the embeddings are
    returned as tensors. This is required to backpropagate through the model head.
    """

    def __init__(
        self, model_body: SentenceTransformer = None, model_head: torch.nn.Module = None
    ):
        """
        model_body: SentenceTransformer(), see huggingface library.
        model_head: nn.Module(), torch implementation of a model head.
                    It should implement .predict(), .predict_proba(), .fit(),
                    same as the sklearn model API.
        """

        super(SetFitModel, self).__init__()
        self.model_body = model_body
        self.model_head = model_head

    def predict(self, x_test):
        embeddings = self.model_body.encode(x_test, convert_to_tensor=True)
        return self.model_head.predict(embeddings)

    def predict_proba(self, x_test):
        embeddings = self.model_body.encode(x_test, convert_to_tensor=True)
        return self.model_head.predict_proba(embeddings)

    def __call__(self, inputs):
        output = self.model_body(inputs)
        embeddings = output["sentence_embedding"]
        return self.model_head.predict_proba(embeddings), output


class SetFitGrad:
    """
    This class takes a SetFit model and deconstructs its operations to
    allow for exploration of gradients.

    Essentially, instead of passing a sentence and getting the probability of a class,
    we can pass a token embedding tensor and do the same + return the gradient w.r.t to
    each token embedding dimension.

    NOTE: This assumes we are interested in a binary classification problem.
    """

    def __init__(self, model: SetFitModel, tokenizer=None, device: int = None):
        self.model_body = model.model_body
        self.model_head = model.model_head

        if device:
            self.device = device
        else:
            self.device = self.model_body.device

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = lambda x: self.model_body.tokenizer(
                x, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

        transformer = self.model_body._modules["0"]._modules["auto_model"]
        self.embedder = transformer._modules["embeddings"]
        self.encoder = transformer._modules["encoder"]

        rest_of_processing = OrderedDict(
            {
                key: value
                for key, value in self.model_body._modules.items()
                if key != "0"
            }
        )
        self.rest_of_processing = Sequential(rest_of_processing)

    def model_pass(self, sentence_string: str = None, sentence_token_embedding=None):
        """
        A SetFit model pass, but broken down into steps.

        Either pass the sentence_string or the sentence_token_embedding.

        Returns:
            "positive_class_probability",
            "token_embedding_gradients",
            "sentence_token_embedding",
            "attention_mask",
            "input_ids",
            "sentence_embedding",

        """

        if sentence_token_embedding is None:
            with torch.no_grad():
                sentence = self.tokenizer(sentence_string).to(device=self.device)
                sentence_token_embedding = self.embedder(
                    input_ids=sentence["input_ids"],
                    token_type_ids=sentence["token_type_ids"],
                )

            attention_mask = sentence["attention_mask"]

            sentence_token_embedding.requires_grad = True
            input_ids = sentence["input_ids"]
        else:
            input_ids = None
            attention_mask_dim = sentence_token_embedding.shape[1:2]
            attention_mask = torch.ones(attention_mask_dim, device=self.device)

        encoder_output = self.encoder(
            sentence_token_embedding, attention_mask=attention_mask
        )

        features = {}
        features["token_embeddings"] = encoder_output[0]
        features["attention_mask"] = attention_mask
        features["input_ids"] = input_ids

        results = self.rest_of_processing(features)

        positive_class_probability = self.model_head.predict_proba(
            results["sentence_embedding"]
        )

        token_embedding_gradients = grad(
            outputs=positive_class_probability,
            inputs=sentence_token_embedding,
            grad_outputs=torch.ones_like(positive_class_probability),
            retain_graph=True,
        )[0].squeeze()

        output = namedtuple(
            "SetFitGrad",
            [
                "positive_class_probability",
                "token_embedding_gradients",
                "sentence_token_embedding",
                "attention_mask",
                "input_ids",
                "sentence_embedding",
            ],
        )

        return output(
            positive_class_probability,
            token_embedding_gradients,
            sentence_token_embedding,
            attention_mask,
            input_ids,
            results["sentence_embedding"],
        )
