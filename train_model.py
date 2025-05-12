# train_model.py
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    data_dir = 'datasets'
    all_data = []

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, header=None)

            # Check if first column is the label, and rest are landmarks
            if df.shape[1] == 64:  # 1 label + 21 landmarks * 3 = 63 features
                df.columns = ['label'] + [f'f{i}' for i in range(1, 64)]
                all_data.append(df)
            else:
                print(f"[WARNING] Skipping {file} due to unexpected format.")

    if not all_data:
        print("[ERROR] No valid data found.")
        return

    df = pd.concat(all_data, ignore_index=True)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train SVM
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    # Accuracy
    accuracy = clf.score(X_test, y_test)
    print(f"[INFO] Model trained with accuracy: {accuracy * 100:.2f}%")

    # Save model and label encoder
    os.makedirs('model', exist_ok=True)
    joblib.dump(clf, 'model/svm_model.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
    print("[INFO] Model and encoder saved in 'model/' directory.")

if __name__ == "__main__":
    train_model()
