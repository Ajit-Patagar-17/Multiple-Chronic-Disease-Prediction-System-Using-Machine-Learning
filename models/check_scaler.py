import pickle

scaler = pickle.load(open("models/COPD_scaler.pkl", "rb"))

try:
    print(scaler.feature_names_in_)
except:
    print("Scaler does NOT contain feature_names_in_. It was trained on a numpy array!")
