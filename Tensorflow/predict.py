from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd


def dataframe_loader(product_file_directory, user_file_directory, rating_file_directory):
    # products items
    product = pd.read_csv(product_file_directory)
    # product = product.columns
    product = product.loc[:, ["product_id", "product_name", "category", "brand", "price", "color", "size"]]

    # interactions with products and users
    ratings = pd.read_csv(rating_file_directory)
    ratings = ratings.loc[:, ["user_id", "product_id", "rating"]]

    # users list
    users = pd.read_csv(user_file_directory)
    users = users.loc[:, ["user_id", "age", "gender", "location", "occupation"]]

    # merging data if required
    # product_rating_data = pd.merge(product, ratings)
    # full_data = pd.merge(product_rating_data, users)
    return product, users, ratings


def recommend(product_file_directory, rating_file_directory, user_file_directory):
    loaded_model = load_model('recommender_model.h5py')
    product, users, ratings = dataframe_loader(product_file_directory, user_file_directory, rating_file_directory)
    # input should be rating dataframe
    user_ids = ratings["user_id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    product_ids = ratings["product_id"].unique().tolist()
    product2product_encoded = {x: i for i, x in enumerate(product_ids)}
    product_encoded2product = {i: x for i, x in enumerate(product_ids)}
    ratings["user"] = ratings["user_id"].map(user2user_encoded)
    ratings["product"] = ratings["product_id"].map(product2product_encoded)
    product_df = product
    # Let us get a user and see the top recommendations.
    user_id = ratings.user_id.sample(1).iloc[0]
    products_watched_by_user = ratings[ratings.user_id == user_id]
    products_not_watched = product_df[
        ~product_df["product_id"].isin(products_watched_by_user.product_id.values)
    ]["product_id"]
    products_not_watched = list(
        set(products_not_watched).intersection(set(product2product_encoded.keys()))
    )
    products_not_watched = [[product2product_encoded.get(x)] for x in products_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_product_array = np.hstack(
        ([[user_encoder]] * len(products_not_watched), products_not_watched)
    )
    ratings = loaded_model.predict(user_product_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_product_ids = [
        product_encoded2product.get(products_not_watched[x][0]) for x in top_ratings_indices
    ]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("products with high ratings from user")
    print("----" * 8)
    top_products_user = (
        products_watched_by_user.sort_values(by="rating", ascending=False)
        .head(5)
        .product_id.values
    )
    product_df_rows = product_df[product_df["product_id"].isin(top_products_user)]
    for row in product_df_rows.itertuples():
        print(row.product_name, ":", row.category)

    print("----" * 8)
    print("Top 10 product recommendations")
    print("----" * 8)
    recommended_products = product_df[product_df["product_id"].isin(recommended_product_ids)]
    for row in recommended_products.itertuples():
        print(row.product_name, ":", row.category)


if __name__ == "__main__":
    product_file = "./product.csv"
    ratings_file = "./interactions.csv"
    users_file = "./user.csv"
    recommend(product_file, ratings_file, users_file)
