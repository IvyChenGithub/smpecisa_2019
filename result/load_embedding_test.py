import pickle

embedding_path = r"./data/embedding1.pkl"
with open(embedding_path, "rb") as embedding_file:
    embedding = pickle.load(embedding_file)
    # embedding = pickle.load(open(embedding_path, 'rb'))
    print(embedding["王室"])
    print(type(embedding["王室"]))
    print(embedding["王室"].shape)
    print(embedding.keys())
    keys = list(embedding.keys())
    print(len(keys))
