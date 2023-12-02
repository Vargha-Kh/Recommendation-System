import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD, accuracy, KNNBasic
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate


class ContentRecommender:
    def __init__(self, product_file_directory, user_file_directory, rating_file_directory):
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

    # Create a function to recommend products based on user preferences
    def content_based_recommendations(self, product_title, top_n=10):
        product_data = self.product
        # Create a TF-IDF vectorized to convert product genres into numerical features
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        product_genres_matrix = tfidf_vectorizer.fit_transform(product_data['category'])

        # Calculate the cosine similarity between products based on their genres
        cosine_sim = linear_kernel(product_genres_matrix, product_genres_matrix)
        # Find the index of the product with the given title
        product_idx = product_data.index[product_data['product_name'] == product_title].tolist()[0]

        # Calculate the cosine similarity scores for all products
        sim_scores = list(enumerate(cosine_sim[product_idx]))

        # Sort the products based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top N most similar products
        sim_scores = sim_scores[1:top_n + 1]

        # Get the product indices
        product_indices = [i[0] for i in sim_scores]

        # Return the top N similar products
        return product_data['product_name'].iloc[product_indices]

    def category_filtering(self, product_id, num_recommendation=10):
        # Extracting the genres column
        genres = self.product['category']

        # Creating an instance of the OneHotEncoder
        encoder = OneHotEncoder()

        # Fitting and transforming the genres column
        genres_encoded = encoder.fit_transform(genres.values.reshape(-1, 1))

        # Creating an instance of the NearestNeighbors class
        recommender = NearestNeighbors(metric='cosine')

        # Fitting the encoded genres to the recommender
        recommender.fit(genres_encoded.toarray())

        # Index of the product the user has previously watched
        product_index = product_id

        # Number of recommendations to return
        num_recommendations = num_recommendation

        # Getting the recommendations
        _, recommendations = recommender.kneighbors(genres_encoded[product_index].toarray(),
                                                    n_neighbors=num_recommendations)
        # Extracting the product titles from the recommendations
        return self.product.iloc[recommendations[0]]['product_name']


class CollaborativeRecommender:
    def __init__(self, product_file_directory, user_file_directory, rating_file_directory):
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
        # product_rating_data = pd.merge(self.product, self.ratings)
        # self.full_data = pd.merge(product_rating_data, self.users)

    # Collaborative filtering model
    def train_collaborative_filtering_model(self, user_ratings):
        # Load data and create a Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(user_ratings[['user_id', 'product_id', 'rating']], reader)

        # Split data into training and testing sets
        trainset, testset = train_test_split(data, test_size=0.2)

        # Initialize and train the SVD algorithm (or another collaborative filtering model)
        algo = KNNBasic()
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
        algo.fit(trainset)

        predictions = algo.test(testset, verbose=False)

        # Then compute RMSE
        accuracy.rmse(predictions, verbose=False)

        return algo

    def get_collaborative_filtering_recommendations(self, model, user_id, top_n=10):
        # Generate recommendations using the trained model
        # We can use the model to predict ratings for items not rated by the user
        # and recommend items with the highest predicted ratings.
        # Here's a simplified example using Surprise's built-in functions:

        # Create a list of product IDs that the user has not rated
        unrated_product_ids = [product_id for product_id in range(200) if
                               not self.user_has_rated_product(user_id, product_id, self.ratings)]

        # Predict ratings for unrated products
        predictions = [model.predict(user_id, product_id) for product_id in unrated_product_ids]

        # Sort predictions by predicted rating in descending order
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get the top N recommended product IDs
        top_product_ids = [prediction.iid for prediction in predictions[:top_n]]

        return top_product_ids

    def user_has_rated_product(self, user_id, product_id, user_ratings):
        # Helper function to check if a user has rated a specific product
        return not user_ratings[(user_ratings['user_id'] == user_id) & (user_ratings['product_id'] == product_id)].empty

    def collaborative_recommend(self, user_id):
        model = self.train_collaborative_filtering_model(self.ratings)
        output_id = self.get_collaborative_filtering_recommendations(model, user_id)
        print(output_id)
        for id_num in output_id:
            id_num = id_num + 1
            print(id_num)
            product_name = self.product.loc[self.product['product_id'] == id_num, 'product_name'].iloc[0]
            print(product_name)


if __name__ == '__main__':
    # Example usage: Get recommendations for a specific product
    product_file = "./product.csv"
    ratings_file = "./interactions.csv"
    users_file = "./user.csv"

    content_recommender = ContentRecommender(product_file, users_file, ratings_file)
    print("Recommendation Based on Item: ")
    recommendations = content_recommender.content_based_recommendations("T-shirt_Garcia-Walker_4")
    print(recommendations)
    print("\nRecommendation Based on Category: ")
    recommended_product_titles = content_recommender.category_filtering(product_id=5)
    print(recommended_product_titles)
    print("\nRecommendation Based on User: ")
    collab_recommender = CollaborativeRecommender(product_file, users_file, ratings_file)
    collab_recommender.collaborative_recommend(user_id=2)