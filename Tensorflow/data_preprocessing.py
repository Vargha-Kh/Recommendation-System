import pandas as pd
import numpy as np


class CustomTFDataset:
    def __init__(self):
        self.users = None
        self.ratings = None
        self.product = None

    def dataframe_loader(self, product_file_directory, user_file_directory, rating_file_directory):
        # products items
        self.product = pd.read_csv(product_file_directory)
        # self.product = self.product.columns
        self.product = self.product.loc[:, ["product_id", "product_name", "category", "brand", "price", "color", "size"]]

        # interactions with products and users
        self.ratings = pd.read_csv(rating_file_directory)
        # self.ratings - self.ratings.columns
        self.ratings = self.ratings.loc[:, ["user_id", "product_id", "rating"]]

        # users list
        self.users = pd.read_csv(user_file_directory)
        # self.users = self.users.columns
        self.users = self.users.loc[:, ["user_id", "age", "gender", "location", "occupation"]]

        # merging data if required
        # product_rating_data = pd.merge(product, ratings)
        # full_data = pd.merge(product_rating_data, users)
        return self.product, self.users, self.ratings

    def keras_data_loader(self, product_file_directory, user_file_directory, rating_file_directory):
        product, users, ratings = self.dataframe_loader(product_file_directory, user_file_directory,
                                                        rating_file_directory)
        # input should be rating dataframe
        user_ids = ratings["user_id"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}
        product_ids = ratings["product_id"].unique().tolist()
        product2product_encoded = {x: i for i, x in enumerate(product_ids)}
        product_encoded2product = {i: x for i, x in enumerate(product_ids)}
        ratings["user"] = ratings["user_id"].map(user2user_encoded)
        ratings["product"] = ratings["product_id"].map(product2product_encoded)

        num_users = len(user2user_encoded)
        num_products = len(product_encoded2product)
        ratings["rating"] = ratings["rating"].values.astype(np.float32)
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(ratings["rating"])
        max_rating = max(ratings["rating"])

        print(
            "Number of users: {}, Number of products: {}, Min rating: {}, Max rating: {}".format(
                num_users, num_products, min_rating, max_rating
            )
        )

        ratings = ratings.sample(frac=1, random_state=42)
        x = ratings[["user", "product"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = ratings["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int(0.9 * ratings.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )

        return x_train, x_val, y_train, y_val, (num_users, num_products)
