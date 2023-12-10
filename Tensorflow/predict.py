import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def dataframe_loader(product_file, user_file, rating_file):
    product = pd.read_csv(product_file)[["product_id", "product_name", "category", "brand", "price", "color", "size"]]
    ratings = pd.read_csv(rating_file)[["user_id", "product_id", "rating"]]
    users = pd.read_csv(user_file)[["user_id", "age", "gender", "location", "occupation"]]
    return product, users, ratings


def recommend(product_file, rating_file, user_file, model_path='recommender_model.h5py', top_n=10):
    loaded_model = load_model(model_path)
    product, users, ratings = dataframe_loader(product_file, user_file, rating_file)

    # Encode user and product IDs
    user2user_encoded = {x: i for i, x in enumerate(ratings["user_id"].unique())}
    product2product_encoded = {x: i for i, x in enumerate(ratings["product_id"].unique())}
    userencoded2user = {i: x for i, x in enumerate(ratings["user_id"].unique())}
    productencoded2product = {i: x for i, x in enumerate(ratings["product_id"].unique())}

    ratings["user"] = ratings["user_id"].map(user2user_encoded)
    ratings["product"] = ratings["product_id"].map(product2product_encoded)

    # Get a random user
    user_id = ratings["user_id"].sample(1).iloc[0]

    # Filter watched and unwatched products
    products_watched_by_user = ratings[ratings["user_id"] == user_id]
    products_not_watched = product[~product["product_id"].isin(products_watched_by_user["product_id"].values)]

    # Get recommendations
    user_encoder = user2user_encoded[user_id]
    products_not_watched["user"] = user_encoder
    user_product_array = np.hstack(
        ([[user_encoder]] * len(products_not_watched), products_not_watched["product_id"].values.reshape(-1, 1)))
    ratings = loaded_model.predict(user_product_array).flatten()

    # Top N recommendations
    top_ratings_indices = ratings.argsort()[-top_n:][::-1]
    recommended_product_ids = [productencoded2product[i] for i in top_ratings_indices]

    # Display recommendations
    print("Showing recommendations for user:", user_id)
    print("=" * 30)

    # Top products watched by user
    print("Top products with high ratings from the user:")
    print("-" * 30)
    top_products_user = products_watched_by_user.sort_values(by="rating", ascending=False).head(5)["product_id"].values
    product_df_rows_user = product[product["product_id"].isin(top_products_user)]
    for row in product_df_rows_user.itertuples():
        print(row.product_name, ":", row.category)

    # Recommended products
    print("-" * 30)
    print("Top {} product recommendations:".format(top_n))
    print("-" * 30)
    recommended_products = product[product["product_id"].isin(recommended_product_ids)]
    for row in recommended_products.itertuples():
        print(row.product_name, ":", row.category)


if __name__ == "__main__":
    product_file = "./product.csv"
    ratings_file = "./interactions.csv"
    users_file = "./user.csv"
    recommend(product_file, ratings_file, users_file)
