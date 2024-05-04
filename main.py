import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load


def train_model():
    df = pd.read_csv("Crop_Dataset.csv")
    df = preprocess_data(df)
    X = df.drop(["Label_Encoded"], axis=1)
    y = df["Label_Encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1234)
    rf_model.fit(X_train, y_train)

    # Evaluate and Save
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Model Accuracy:", accuracy_rf)

    # Save the model
    dump(rf_model, 'crop_recommendation_rf_model.joblib')


def preprocess_data(data):
    if "Total_Nutrients" in data.columns and "Temperature_Humidity" in data.columns and "Label" in data.columns and "Log_Rainfall" in data.columns:
        data = data.drop(["Total_Nutrients", "Temperature_Humidity", "Label", "Log_Rainfall"], axis=1)
    data["temperature"] = data["temperature"].round(2)
    data["humidity"] = data["humidity"].round(2)
    data["ph"] = data["ph"].round(2)
    data["rainfall"] = data["rainfall"].round(2)
    scaler = StandardScaler()
    numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    # data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data


def predict_new_data(new_data):
    rf_model = load('crop_recommendation_rf_model.joblib')
    new_data = preprocess_data(new_data)
    probabilities = rf_model.predict_proba(new_data)
    crops_probabilities = list(zip(rf_model.classes_, probabilities[0]))
    sorted_crops = sorted(crops_probabilities, key=lambda x: x[1], reverse=True)
    top_three_crops = sorted_crops[:3]

    print("Recommended Crops:")
    for crop, probability in top_three_crops:
        print(f"- {crop}: Probability = {probability:.2f}")


if __name__ == '__main__':
    train_model()
    sample_data = pd.DataFrame({
        'N': [67],
        'P': [46],
        'K': [44],
        'temperature': [20.555],
        'humidity': [80.2222],
        'ph': [6.6665],
        'rainfall': [150.8963],
    })
    # predict_new_data(sample_data)
