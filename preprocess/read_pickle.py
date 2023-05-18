import pickle

with open ('../data/sentimental_clauses_en.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)