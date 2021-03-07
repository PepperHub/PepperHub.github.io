---
layout: post
title: Airbnb Seattle Data Exploration
subtitle: Exploratory data analysis to derive the most relevant characteristics that
are correlated to the price of the listings in Seattle
date: 2021-03-07
cover-img: /assets/img/SeattleCity2_pic.jpg
thumbnail-img: /assets/img/seattle_thumb_pic.jpg
share-img: /assets/img/path.jpg
tags: [Airbnb, Seattle]
htmlwidget: true
---

## Introduction

Airbnb is a home-sharing/renting platform for those who would like to whether stay in a flat/villa/bungalow etc. or rent their room/flat for a specified period. Even though there are guidelines on how to set prices on a listing as a host, there is no set rule. Airbnb has a feature called 'smart pricing' that could be used as starting point to set prices for a listing, especially for less experienced hosts.

In this post, we will take a look at the Seattle listings dataset that were scraped on the 25th of October 2020 to answer some of the following questions, which we derived after having a closer look through the data:

- __Question set 1:__ Which features are highly correlated with price?

- __Question set 2:__ Do reviews have any effect on the price? Which factors affect the overall rating? Is it profitable to have the high rating?

- __Question set 3:__ What are the most common property and room types that are listed? How are the prices distributed amongst them?

We have 4335 listings and 74 features, which are not in a clean format. It shows the overall price advertised by the host rather than the average price paid by the guests. However, we will use this dataset as a proof of concept rather than the source of truth. We begin our analysis with data preparation steps to re-format the features that would be informative for building a model and predict the price given the characteristics of the bespoke listings.

## Data preparation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import myutils as myu
import re
%matplotlib inline
listing_df = pd.read_csv('../data/seattle_listings.csv.gz') 
```

We remove all the columns that have the same constant value for the entire dataset and
are completely missing and contain URLs since we are not going to do any type of text
analytics on this project.
