import pandas as pd


class CustomTFDataset:
    def __init__(self):
        self.users = None
        self.ratings = None
        self.product = None

    def dataframe_loader(self, product_file_directory, user_file_directory, rating_file_directory):
        # Load product, users, and ratings dataframes
        self.product = pd.read_csv(product_file_directory)[
            ["product_id", "product_name", "category", "brand", "price", "color", "size"]]
        self.ratings = pd.read_csv(rating_file_directory)[["user_id", "product_id", "rating"]]
        self.users = pd.read_csv(user_file_directory)[["user_id", "age", "gender", "location", "occupation"]]
        return self.product, self.users, self.ratings

    def keras_data_loader(self, product_file_directory, user_file_directory, rating_file_directory):
        product, users, ratings = self.dataframe_loader(product_file_directory, user_file_directory,
                                                        rating_file_directory)

        # Mapping user and product IDs
        user2user_encoded = {user_id: i for i, user_id in enumerate(ratings["user_id"].unique())}
        product2product_encoded = {product_id: i for i, product_id in enumerate(ratings["product_id"].unique())}
        ratings["user"] = ratings["user_id"].map(user2user_encoded)
        ratings["product"] = ratings["product_id"].map(product2product_encoded)

        # Normalize ratings between 0 and 1
        min_rating, max_rating = ratings["rating"].min(), ratings["rating"].max()
        ratings["rating"] = (ratings["rating"] - min_rating) / (max_rating - min_rating)

        # Shuffle and split data into training and validation sets
        ratings = ratings.sample(frac=1, random_state=42)
        x = ratings[["user", "product"]].values
        y = ratings["rating"].values
        train_indices = int(0.9 * ratings.shape[0])
        x_train, x_val, y_train, y_val = x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]

        num_users, num_products = len(user2user_encoded), len(product2product_encoded)

        print(
            f"Number of users: {num_users}, Number of products: {num_products}, Min rating: {min_rating}, Max rating: {max_rating}")

        return x_train, x_val, y_train, y_val, (num_users, num_products)


if __name__ == "__main__":
    product_file = "./product.csv"
    ratings_file = "./interactions.csv"
    users_file = "./user.csv"

    dataset = CustomTFDataset()
    x_train, x_val, y_train, y_val, input_shape = dataset.keras_data_loader(product_file, users_file, ratings_file)
