#!/usr/bin/env python
# coding: utf-8

# # Quick demo of setfit_explanations

# In[1]:


# In[1]:


import torch

import pandas as pd

from setfit import SetFitModel, SetFitTrainer

from setfit_explanations.integrated_gradients import integrated_gradients_on_text
from setfit_explanations.setfit_extensions import SetFitGrad, SetFitModelWithTorchHead
from setfit_explanations.model_head import BinaryLogisticRegressionModel
from setfit_explanations.html_text_colorizer import WordImportanceColorsSetFit

from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.SentenceTransformer import SentenceTransformer

from tqdm.auto import tqdm

from IPython.display import HTML


# In[2]:


from datasets import load_dataset

data = load_dataset("rotten_tomatoes")
data = data["train"].train_test_split(
    train_size=20, test_size=300, stratify_by_column="label", shuffle=True
)


# In[3]:


data = {
    "train": {
        "text": ["happy", "sad"],
        "label": [1, 0],
    },
}


# In[4]:


# train = data["train"]
# test = data["test"]
train = data["train"]
test = data["train"]


# Warning: this is a big model
# model_name = "sentence-transformers/all-MiniLM-L12-v2"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SetFitModelWithTorchHead(
    model_body=SentenceTransformer(model_name),
    model_head=BinaryLogisticRegressionModel(
        input_dimension=384, lr=0.01, number_of_epochs=10000, device="cpu"
    ),
)

trainer = SetFitTrainer(
    model=model,
    train_dataset=train,
    eval_dataset=test,
    loss_class=CosineSimilarityLoss,
    batch_size=10,
    num_epochs=2,
    num_iterations=20,  # for contrastive learning
)


# In[6]:


trainer.train()


# In[7]:


grd = SetFitGrad(model)
m = WordImportanceColorsSetFit(grd)


# In[8]:
N = 0
test_text, test_label = test["text"][N], test["label"][N]
colors, df, prob = m.show_colors_for_sentence(test_text, integration_steps=100)
print(test_label)
print(f"class probability: {prob:1.2f}")
print(df)
HTML(colors)

N = 1
test_text, test_label = test["text"][N], test["label"][N]
colors, df, prob = m.show_colors_for_sentence(test_text, integration_steps=100)
print(test_label)
print(f"class probability: {prob:1.2f}")
print(df)
HTML(colors)


# In[42]:


df


# In[ ]:
