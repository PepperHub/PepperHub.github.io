---
layout: post
title: Airbnb Seattle Data Exploration
subtitle: Exploring the listings in Seattle
cover-img: /assets/img/airbnb-data-exploration/SeattleCity2_pic.jpg
thumbnail-img: /assets/img/airbnb-data-exploration/seattle_thumb_pic.jpg
share-img: /assets/img/airbnb-data-exploration/SeattleCity2_pic.jpg
tags: [Airbnb, Seattle]
---

## Introduction

Airbnb is a home-sharing platform for those who would like to whether stay in a flat/villa/bungalow etc. or rent their property for a specified period. Even though there are guidelines on how to set prices on a listing as a host, there is no set rule. Airbnb has a feature called 'smart pricing' that could be used as starting point to set prices for a listing, especially for less experienced hosts.

In this post, we will take a look at the Seattle listings dataset that were scraped on the 25th of October 2020 to answer some of the following questions, which we derived after having a closer look through the data:

- __Question set 1:__ Which features are highly correlated with the price?

- __Question set 2:__ Do reviews have any effect on the price? Which factors affect the overall rating? Is it profitable to have a high rating?

- __Question set 3:__ What are the most common property and room types that are listed? How are the prices distributed amongst them?

We have 4335 listings and 74 features, which are not in a clean format. It only includes the overall price advertised by the host rather than the average price paid by the guests. However, we will use this dataset as a proof of concept rather than the source of truth. We begin our analysis with data preparation steps to re-format the features that would be informative for our exploratory analysis.

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

We remove all the columns that have the same constant value for the entire dataset or
are completely missing or contain URLs since we are not going to do any type of text
analytics on this project.

```python
remove_list = listing_df.columns[(listing_df.nunique()<=1)
                                 | (listing_df.columns.str.contains('url'))
                                 | (listing_df.isnull().mean()==1)
                                ].tolist()                                
remove_list
```

```python
## ['listing_url',
##  'scrape_id',
##  'last_scraped',
##  'picture_url',
##  'host_url',
##  'host_thumbnail_url',
##  'host_picture_url',
##  'bathrooms',
##  'calendar_updated',
##  'has_availability',
##  'calendar_last_scraped']
 ```

We make sure all date columns are in date format. We create a feature that shows the number of days the hosts listed their properties. The data was scraped on the 2020-10-25, so we will calculate the number of days the hosts listed their properties until the date the scraped date. Similarly, we create a column stating the number of days since the listing last reviewed.

```python
listing_df["days_since_host"] =  (listing_df['last_scraped'] - listing_df['host_since']).dt.days
listing_df["days_since_last_review"] =  (listing_df['last_scraped'] - listing_df['last_review']).dt.days
listing_df["days_since_first_review"] =  (listing_df['last_scraped'] - listing_df['first_review']).dt.da
listing_df[["days_since_host","days_since_last_review", "days_since_first_review"]].head()
```
<br/>
 | days_since_host | days_since_last_review number | days_since_first_review number |
 | :------ |:--- | :--- |
 | 4443 | 267 | 4423 |
 | 4095 | 303 | 3740 |
 | 4091 | 300 | 3209 |
 | 4091 | 55 | 3206 |
 | 4200 | 27 | 3420 |

We remove special characters from numeric columns such as % or texts.

```python
col_list = ["price", "host_response_rate", "host_acceptance_rate"]
listing_df[col_list] = listing_df[col_list].apply(lambda x: remove_special_char(x, col_list), axis = 1)
```

We create more useful variables from the _bathrooms_text_ variable by following the
steps below:

- Create a new variable called _private_bath_yn_, which indicates whether a bath is private or shared.
- Replace the values contain '-bath' and are half bath with 0.5.
- Remove all non-digit characters.
- Fill empty values with 0.
- Assign it to a new variable called _n_bathrooms_.

Here are the unique values of _bathrooms_text_ variable:

```python
listing_df.bathrooms_text.unique()
## array(['2.5 baths', '3 shared baths', '1 bath', '2 shared baths',
##        '2 baths', '1 private bath', '1.5 baths', '4 shared baths',
##        '1 shared bath', '3 baths', '0 baths', '1.5 shared baths',
##        '4.5 baths', nan, '3.5 baths', 'Half-bath', '5 shared baths',
##        '2.5 shared baths', '4 baths', '3.5 shared baths',
##        '0 shared baths', '6 baths', 'Shared half-bath',
##        'Private half-bath'], dtype=object)
```
<br/>
```python
listing_df['private_bath_yn'] = np.where(listing_df.bathrooms_text.str.contains('private'), 1, 0)
listing_df['n_bathrooms'] = listing_df.bathrooms_text.str.replace(r'(^.*-bath.*$)', '0.5', regex=True)
listing_df['n_bathrooms'] = listing_df.n_bathrooms.str.replace("[^\d\.]", "", regex = True).fillna(0)
```

Below is an example of the converted variables that will be used in our explanatory analysis.

```python
listing_df[['private_bath_yn', 'bathrooms_text', 'n_bathrooms']].head(5)
```
<br/>
| private_bath_yn | bathrooms_text number | n_bathrooms number |
| :---------- | :---------- | :---------- |
| 0 |	2.5 baths |	2.5 |
| 0	| 3 shared baths | 3 |
| 0 | 1 bath | 1 |
| 0 | 1 bath | 1 |
| 0 | 1 bath | 1 |

We calculate the estimated revenue using the minimum nights a guest would stay in the host's accommodation.

```python
listing_df['estimated_revenue'] = listing_df['price'] * listing_df['minimum_nights']
```

You can find the details about how we cleaned and reformated all the other variables [here](https://github.com/PepperHub/airbnb_seattle_price_prediction/blob/main/venv/analysis_notebook.ipynb).

### Question 1

- Which features are highly correlated with each other and/or with the price?

<img src="/assets/img/airbnb_data_exploration/heatmap.png" width="900px" />

The number of people that the property can accommodate, the number of bedrooms, and the number of beds are highly correlated with each other. Although they are also highly correlated with the price (our target variable), to reduce the multicollinearity, we will remove beds and bedrooms from the dataset. The number of people who can be accommodated would be more relevant to the guests than the other features.
host total listings count and host listing count variables are correlated with calculated host listings count. We will keep the calculated host listings count. variable and drop the others.

### Question 2

- Do reviews have any effect on the price?
- Which factors affect the overall rating? Is it profitable to have a high rating?

```python
(listing_df[['number_of_reviews','review_scores_rating', 'minimum_nights','accommodates','estimated_revenue']]
 .query("number_of_reviews > 0")
 .sort_values('estimated_revenue', ascending=False).head())
 ```
<br/>
| number_of_reviews	| review_scores_rating |	minimum_nights|	accommodates | estimated_revenue |
| :------ |:------| :------ | :------ |
| 10 | 70.0 | 100 | 5 | 90000.0 |
| 11	| 91.0 | 	30 | 	6	| 45000.0 |
| 69 | 96.0	| 30	| 3	| 39120.0 |
| 25 | 93.0	30 | 	8	| 32370.0 |
| 22 | 	99.0 | 182	| 4	| 29120.0 |


```python
(listing_df[['number_of_reviews','review_scores_rating', 'minimum_nights','accommodates','estimated_revenue']]
 .query("number_of_reviews > 0 and minimum_nights <30")
 .sort_values('estimated_revenue', ascending=False).head())
```
<br/>
| number_of_reviews	| review_scores_rating |	minimum_nights|	accommodates | estimated_revenue |
| :------ |:------| :------ | :------ |
| 20	| 99.0	| 29 | 	12 | 	17081.0	|
| 70 | 96.0	| 25 |	7	| 6775.0 |
| 19	| 99.0 | 10	| 6 |	6750.0 |
| 3	| 100.0 |	10 | 8 |	6500.0 |
| 287 | 100.0 | 28 | 6 | 6160.0 |

<br/>
```python
(listing_df[['minimum_nights','number_of_reviews','estimated_revenue','review_scores_rating']]
 .query("number_of_reviews > 0")
 .corr())
```
<br/>
| minimum_nights	| number_of_reviews |	estimated_revenue|	review_scores_rating |
| :--------- |:--------- |:--------- |:--------- |
| minimum_nights	| 1.000000 | -0.187925 | 0.641146 |	-0.080721 |
| number_of_reviews | -0.187925	| 1.000000 | -0.152691 | 0.159541 |
| estimated_revenue	| 0.641146 | -0.152691 | 1.000000	| -0.037708 |
| review_scores_rating	| -0.080721 |	0.159541 | -0.037708 | 1.000000 |

Looking at the top 5 revenue listings, we see that the top earners host their place for at least a month. And short term listings have lower revenue than long-term listings.

Revenue is correlated with the number of nights the listing is advertised. However, score rating and the number of reviews do not seem to have any impact on the revenue of the listing.

### Question 3

What are the most common property and room types that are listed? How are the prices distributed amongst them?

<img src="/assets/img/airbnb_data_exploration/property_type_violin_plt.png" width="700px" />

<br/>

<img src="/assets/img/airbnb_data_exploration/property_type_bar_chart.png" width="700px" />


By looking at the prices of property types, on average, staying in boats seem to be the most expensive option and the private room in a guest suite is the cheapest option on these listings. The variation in house prices is higher than other property types. In terms of the popularity though, the most popular listing property types are apartments and houses in Seattle.

## Conclusion

In this article, we had a look at which characteristics affect the price listings set and the estimated revenue made by the hosts in Seattle. Equally, many other questions could be explored around the features of the listings such as amenities, area, etc. Here we focused on data cleaning and exploratory analysis only to answer set of questions.

Our key takeaways from the questions we explored were:

- Higher the number of people a listing can accommodate, the more likely it that the host will make higher estimated revenue.
- Surprisingly, ratings and the number of reviews do not have much impact on the estimated revenue. However, it is still nice to give a review for future hosts to get a better feel of the listing.
- Apartments and houses are the most common property types in these listings. The most expensive option turned out to be boat listings.
