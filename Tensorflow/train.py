from model import RecommenderNet
import tensorflow as tf
from tensorflow import keras
from data_preprocessing import CustomTFDataset
import matplotlib.pyplot as plt


def recommender_net_training(x_train, x_val, y_train, y_val, num_users, num_products, EMBEDDING_SIZE=50):
    model = RecommenderNet(num_users, num_products, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=512,
        epochs=30,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    # Save the trained model to a file
    model.save('recommender_model.h5py', save_format='tf')

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    keras_dataloader = CustomTFDataset()
    product_file = "./product.csv"
    ratings_file = "./interactions.csv"
    users_file = "./user.csv"
    x_train, x_val, y_train, y_val, (num_users, num_products) = keras_dataloader.keras_data_loader(product_file,
                                                                                                   users_file,
                                                                                                   ratings_file)
    recommender_net_training(x_train, x_val, y_train, y_val, num_users, num_products)
