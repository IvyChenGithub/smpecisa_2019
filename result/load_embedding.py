import pickle
embedding_path = r"./data/embedding1.pkl"


def get_embedding_data(file_path=embedding_path):
    with open(file_path, "rb") as embedding_file:
        embedding = pickle.load(embedding_file)
    return embedding
