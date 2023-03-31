# setfit-integrated-gradients

Hacking SetFit so that it works with [integrated gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients). See demo.ipynb for an example.

I wrote this mini-library before SetFit 0.6.0. At the time, there was no SetFitHead class yet, so I just took the sklearn LogisticRegression and passed its parameters to an equivalent Torch class. I did my best to break the forward pass of SetFit into pieces so that I can push gradients through the head and up to the token embeddings.

Attributions from integrated gradients are computed per token and then averages to get word-level attributions.

I'm leaving this here for posterity and in case it is useful to others for further hacking.

## Installation

```
pip install -e . 
```
