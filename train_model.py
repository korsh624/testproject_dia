import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# === 1. Загрузка данных ===
df = pd.read_csv("diabetes_prediction_dataset.csv")

# === 2. Предобработка ===
df = df.fillna(df.median(numeric_only=True))

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

target_col = "diabetes"
X = df.drop(columns=[target_col])
y = df[target_col]

# === 3. Разделение данных ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Масштабирование ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# === 5. Обучение модели ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === 6. Сохранение модели и масштабировщика ===
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("✅ Модель и масштабировщик сохранены!")
