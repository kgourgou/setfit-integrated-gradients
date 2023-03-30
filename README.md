# setfit-integrated-gradients

Hacking SetFit so that it works with [integrated gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients). See demo.ipynb for an example.

I wrote this mini-library before SetFit 0.6.0. At the time, there was no SetFitHead class yet, so I had to create my own SetFit head class and also break apart the SetFit model so that I can push gradients through the head and up to a certain point. 



I'm leaving this here for posterity and in case it is useful for further hacking.

## Installation

```
pip install -e . 
```


