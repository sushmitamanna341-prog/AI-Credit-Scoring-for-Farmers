import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler

# === Load dataset ===
df = pd.read_csv("crop_dataset_cleaned.csv")

# === Encode categorical columns ===
encoders = {}
for col in ['crop_type', 'location_zone']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# === Encode target column (credit_score) ===
le_score = LabelEncoder()
df['credit_score'] = le_score.fit_transform(df['credit_score'])
encoders['credit_score'] = le_score

# === Encode Image Path as a categorical feature ===
le_image = LabelEncoder()
df['Image Path'] = le_image.fit_transform(df['Image Path'])
encoders['image_path'] = le_image

# === Features and target ===
features = ['crop_type', 'location_zone', 'precipitation', 'avg_temperature', 'soil_moisture', 'Image Path','Yield_1','Yield_2','Yield_3']
X = df[features]
y = df['credit_score']

# === Check the distribution of target classes ===
class_counts = y.value_counts()
print("Class distribution of 'credit_score':")
print(class_counts)

# === Remove classes with fewer than 2 samples ===
classes_to_remove = class_counts[class_counts < 2].index
df_filtered = df[~df['credit_score'].isin(classes_to_remove)]

# === Features and target for filtered data ===
X_filtered = df_filtered[features]
y_filtered = df_filtered['credit_score']

# === Scale only numerical features (excluding categorical image path) ===
scaler = StandardScaler()
X_scaled_filtered = scaler.fit_transform(X_filtered.drop(columns=['Image Path']))

# === Oversampling to handle class imbalance ===
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled_filtered, y_filtered)

# === Train Naive Bayes Classifier ===
clf_nb = GaussianNB()
clf_nb.fit(X_resampled, y_resampled)

# === Evaluate the model ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled_filtered, y_filtered, test_size=0.2, stratify=y_filtered)
y_pred = clf_nb.predict(X_test)

unique_classes = np.unique(y_test)

print("\nNaive Bayes Classification Report:\n")
print(classification_report(y_test, y_pred, labels=unique_classes, target_names=[str(cls) for cls in unique_classes], zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOverall Accuracy: {accuracy:.4f}")
print(f"Overall Weighted F1 Score: {f1:.4f}")

# === Save model and encoders ===
joblib.dump(clf_nb, 'naive_bayes_classifier.pkl')
joblib.dump(scaler, 'scaler_nb.pkl')  # different name to avoid confusion

for col, enc in encoders.items():
    joblib.dump(enc, f'{col.lower().replace(" ", "_")}_encoder.pkl')

print("\nNaive Bayes model and encoders saved successfully.")
