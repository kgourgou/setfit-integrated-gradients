from typing import List, Tuple

import numpy
import pandas as pd
import torch
from scipy.special import roots_legendre
from tqdm import tqdm

from .setfit_extensions import SetFitGrad


def integrated_gradients_on_text(
    sentence_string: str, grd: SetFitGrad, integration_steps: int = 100
) -> Tuple[pd.DataFrame, float, torch.Tensor]:
    """
    Implementation of integrated gradients method for attribution.

    Returns a dataframe with each word in "text" alongside their importance to the
    class label being 1. Essentially a bit more refined than calculating the gradient
    of the probability with respect to each word. Positive gradients indicate words that
    would increase the probability and negative gradients the opposite.

    """

    device = grd.model_head.device

    prob, _, embeddings, _, input_ids, _ = grd.model_pass(
        sentence_string=sentence_string
    )

    init_embed = torch.zeros_like(embeddings, device=device)

    # don't zero out [CLS] and [SEP] tokens
    init_embed[0, 0, :] = embeddings[0, 0, :]
    init_embed[0, -1, :] = embeddings[0, -1, :]

    target_embed = embeddings

    (
        final_scores,
        grad_per_integration_step,
    ) = calculate_integrated_gradient_scores(
        grd, integration_steps, init_embed, target_embed
    )

    scores = pd.DataFrame(
        {
            "token": grd.model_body.tokenizer.convert_ids_to_tokens(
                input_ids.squeeze()
            ),
            "token_ids": input_ids.squeeze().to("cpu"),
            "attribution_score": final_scores,
        }
    ).set_index("token_ids")

    word_to_ids = construct_word_to_id_mapping(sentence_string, grd)

    word_to_score = []

    for word, token_ids in word_to_ids:
        key_scores = scores.loc[token_ids, "attribution_score"].sum()
        word_to_score.append((word, key_scores))

    df_w2s = (
        pd.DataFrame(word_to_score).rename(columns={0: "words", 1: "score"}).dropna()
    )

    return df_w2s, prob, grad_per_integration_step


def calculate_integrated_gradient_scores(
    grd: SetFitGrad,
    integration_steps: int,
    init_embed: torch.Tensor,
    target_embed: torch.Tensor,
    max_alpha: float = 1.0,
):
    """
    grd: SetFitGrad
    integration_steps: int
    init_embed: torch.Tensor, 1 x number of tokens x embedding size
    target_embed: torch.Tensor, 1 x number of tokens x embedding size
    max_alpha: float, up to where to estimate the integral of the gradient curve.
    """
    device = grd.model_head.device

    grads = []

    integration_steps, weights = roots_legendre(integration_steps)

    # originally the steps are in (-1,1), need to map to (0,1)
    integration_steps = numpy.interp(integration_steps, (-1, 1), (0, max_alpha))
    integration_steps = torch.tensor(integration_steps, device=device)

    # scale the weights to the new interval
    weights = torch.tensor(weights, device="cpu") * max_alpha / 2.0

    # function to integrate
    def model_grad_at_perturbation(t):
        new_embed = init_embed + t * (target_embed - init_embed)
        _, gradient, _, _, _, _ = grd.model_pass(sentence_token_embedding=new_embed)
        return gradient

    # TODO could probably do this in one pass instead of using a for-loop.
    for t, w in tqdm(list(zip(integration_steps, weights))):
        gradient = model_grad_at_perturbation(t).cpu().detach()
        diff = (
            target_embed.squeeze().cpu().detach() - init_embed.squeeze().cpu().detach()
        )

        # reduce dimensions from number of tokens x embedding size
        # to number of tokens x 1
        grads.append(gradient * diff * w)

    # integral calculation
    grad_per_integration_step = torch.stack(
        grads
    )  # number of integration steps x number of tokens x embedding dim

    integrals_per_embedding = grad_per_integration_step.sum(
        axis=0
    )  # number of tokens x embedding dim

    final_scores = integrals_per_embedding.mean(axis=1)
    return final_scores, grad_per_integration_step


def construct_word_to_id_mapping(sentence_string, grd) -> Tuple[str, List[int]]:
    word_to_ids = [
        (word, grd.model_body.tokenizer.encode(word, add_special_tokens=False))
        for word in sentence_string.split(" ")
    ]

    return word_to_ids
