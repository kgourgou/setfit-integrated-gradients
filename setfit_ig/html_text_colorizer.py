from typing import Tuple

import numpy
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize, rgb2hex

from .integrated_gradients import integrated_gradients_on_text
from .setfit_extensions import SetFitGrad


def hlstr(string: str, color="white") -> str:
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"


def colorize(attrs, cmap="PiYG"):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """

    # TODO pass an option to have this absolute or relative colouring 
    
    # map colors separately for positive and negative elements
    attrs = attrs.copy()
    pos = attrs[attrs >= 0]
    attrs[attrs >= 0] = numpy.interp(pos, (pos.min(), pos.max()), (0.5, 1))

    neg = attrs[attrs < 0]
    attrs[attrs < 0] = numpy.interp(neg, (neg.min(), neg.max()), (0, 0.5))

    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: rgb2hex(cmap(norm(x))), attrs))
    return colors


class WordImportanceColorsSetFit:
    """
    Helper class to quickly colorize sentences based on SetFitGrad.

    TODO this class could probably be altered to work with other scores and
    other attribution methods.
    """

    def __init__(self, scorer: SetFitGrad):
        self.scorer = scorer
        print("Remember to use:\nfrom IPython.display import HTML")
        print("HTML(colored_text)")

    def show_colors_for_sentence(
        self, text: str, integration_steps: int = 100, cmap: str = "bwr"
    ) -> Tuple[str, pd.DataFrame, float, numpy.array]:
        """
        Pass the output of this function to IPython.display.HTML

        from IPython.display import HTML
        HTML(colored_text)


        """

        df_w2s, prob, grad_per_integration_step = integrated_gradients_on_text(
            text, self.scorer, integration_steps=integration_steps
        )

        colors = colorize(df_w2s.score, cmap=cmap)

        return (
            " ".join(list(map(hlstr, df_w2s.words, colors))),
            df_w2s,
            float(prob.detach()),
            grad_per_integration_step,
        )
