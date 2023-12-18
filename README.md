# Recommendation System
This repository contains an implementation of a Recommendation System, designed to provide personalized recommendations based on user preferences and historical interactions.

# Table of Contents
- Introduction
- Features
- Installation
- Usage
  - Content-Based Recommendations
  - Category Filtering
  - Collaborative Filtering
  - Deep Recommendation system
- Examples

<br/>

## Introduction
The Recommendation System is a tool developed for suggesting items, products, or content to users, aiming to enhance their experience and engagement with a platform. The system utilizes [provide brief information on the underlying algorithm or approach used].

The Sklearn Recommendation System comprises three types of recommenders:

- ContentRecommender: Utilizes TF-IDF vectorization and cosine similarity to provide content-based recommendations and category filtering.

- CollaborativeRecommender: Implements collaborative filtering using the Surprise library to predict user ratings and generate collaborative recommendations.

- Category Filtering: Category filtering is a technique used in recommendation systems to provide recommendations based on the similarity of product categories. The algorithm involves encoding product categories and using a similarity metric, such as cosine similarity, to identify products with similar categories.

The Tensorlow Recommendation system use Neural network for training from datasets:
- Deep Recommendation Algorithms leverage deep learning techniques to model complex patterns and representations in user-item interactions for more accurate and personalized recommendations.

<br/>

## Features
[List key features and capabilities of the Recommendation System.]
[E.g., Collaborative Filtering, Content-Based Filtering, Hybrid Approaches, etc.]

<br/>

## Installation
To install the Recommendation System, follow these steps:

Clone the repository or download the latest release.
[Provide any additional installation steps or dependencies if needed.]

```
pip install -r requirements.txt
```
<br/>

## Usage
### Sklearn: content-based_recommend.py
#### Content-Based Recommendations
To get content-based recommendations based on a product title:

```
content_recommender = ContentRecommender(product_data, user_ratings)
recommendations = content_recommender.content_based_recommendations("T-shirt_Garcia-Walker_4")
print(recommendations)
```

<br/>

#### Category Filtering
To filter products based on category similarity:

```
content_recommender = ContentRecommender(product_data, user_ratings)
recommended_product_titles = content_recommender.category_filtering(product_id=5)
print(recommended_product_titles)
```

<br/>

#### Collaborative Filtering
To perform collaborative filtering and recommend products for a user:

```
collab_recommender = CollaborativeRecommender(product_data, user_ratings)
collab_recommender.collaborative_recommend(user_id=2)
```

<br/>

### Tensorflow: Recommendation Net
#### training process: specify the directory of product_file, users_file, ratings_file and start the training.

```
python Tensorflow/train.py
```

#### inference process: specify the directory of product_file, users_file, ratings_file and start the training.

```
python Tensorflow/predict.py
```


