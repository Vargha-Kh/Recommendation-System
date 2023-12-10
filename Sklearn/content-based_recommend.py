import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate


class BaseRecommender:
    def __init__(self, product_data, user_ratings):
        self.product_data = product_data
        self.user_ratings = user_ratings

    def user_has_rated_product(self, user_id, product_id):
        return not self.user_ratings[(self.user_ratings['user_id'] == user_id)
                                     & (self.user_ratings['product_id'] == product_id)].empty


class ContentRecommender(BaseRecommender):
    def __init__(self, product_data, user_ratings):
        super().__init__(product_data, user_ratings)

    def content_based_recommendations(self, product_title, top_n=10):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        product_genres_matrix = tfidf_vectorizer.fit_transform(self.product_data['category'])
        cosine_sim = linear_kernel(product_genres_matrix, product_genres_matrix)

        product_idx = self.product_data.index[self.product_data['product_name'] == product_title].tolist()[0]

        sim_scores = list(enumerate(cosine_sim[product_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        product_indices = [i[0] for i in sim_scores]

        return self.product_data['product_name'].iloc[product_indices]

    def category_filtering(self, product_id, num_recommendation=10):
        genres = self.product_data['category']
        encoder = OneHotEncoder()
        genres_encoded = encoder.fit_transform(genres.values.reshape(-1, 1))

        recommender = NearestNeighbors(metric='cosine')
        recommender.fit(genres_encoded.toarray())

        product_index = product_id
        num_recommendations = num_recommendation

        _, recommendations = recommender.kneighbors(genres_encoded[product_index].toarray(),
                                                    n_neighbors=num_recommendations)

        return self.product_data.iloc[recommendations[0]]['product_name']


class CollaborativeRecommender(BaseRecommender):
    def __init__(self, product_data, user_ratings):
        super().__init__(product_data, user_ratings)

    def train_collaborative_filtering_model(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.user_ratings[['user_id', 'product_id', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2)

        algo = KNNBasic()
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
        algo.fit(trainset)

        predictions = algo.test(testset, verbose=False)
        accuracy.rmse(predictions, verbose=False)

        return algo

    def get_collaborative_filtering_recommendations(self, model, user_id, top_n=10):
        unrated_product_ids = [product_id for product_id in range(200)
                               if not self.user_has_rated_product(user_id, product_id)]
        predictions = [model.predict(user_id, product_id) for product_id in unrated_product_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_product_ids = [prediction.iid for prediction in predictions[:top_n]]

        return top_product_ids

    def collaborative_recommend(self, user_id):
        model = self.train_collaborative_filtering_model()
        output_id = self.get_collaborative_filtering_recommendations(model, user_id)
        for id_num in output_id:
            id_num = id_num + 1
            product_name = self.product_data.loc[self.product_data['product_id'] == id_num, 'product_name'].iloc[0]
            print(product_name)


def preprocess_data(product_file, user_file, rating_file):
    product_data = pd.read_csv(product_file).loc[:,
                   ["product_id", "product_name", "category", "brand", "price", "color", "size"]]
    user_data = pd.read_csv(user_file).loc[:, ["user_id", "age", "gender", "location", "occupation"]]
    user_ratings = pd.read_csv(rating_file).loc[:, ["user_id", "product_id", "rating"]]
    return product_data, user_data, user_ratings


if __name__ == '__main__':
    product_file = "./product.csv"
    ratings_file = "./interactions.csv"
    users_file = "./user.csv"

    product_data, user_data, user_ratings = preprocess_data(product_file, users_file, ratings_file)

    content_recommender = ContentRecommender(product_data, user_ratings)
    print("Recommendation Based on Item: ")
    recommendations = content_recommender.content_based_recommendations("T-shirt_Garcia-Walker_4")
    print(recommendations)

    print("\nRecommendation Based on Category: ")
    recommended_product_titles = content_recommender.category_filtering(product_id=5)
    print(recommended_product_titles)

    print("\nRecommendation Based on User: ")
    collab_recommender = CollaborativeRecommender(product_data, user_ratings)
    collab_recommender.collaborative_recommend(user_id=2)
