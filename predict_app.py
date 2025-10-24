import joblib
import numpy as np

# === 1. Загрузка модели и вспомогательных объектов ===
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

print("🔮 Программа предсказания диабета")
print("Введите данные по каждому признаку:")

# === 2. Сбор данных с пользователя ===
user_data = []
for feature in feature_names:
    while True:
        try:
            value = float(input(f"{feature}: "))
            user_data.append(value)
            break
        except ValueError:
            print("❌ Введите число!")

# === 3. Преобразование данных ===
user_array = np.array(user_data).reshape(1, -1)
user_scaled = scaler.transform(user_array)

# === 4. Предсказание ===
prediction = model.predict(user_scaled)[0]
prob = model.predict_proba(user_scaled)[0][1]

# === 5. Вывод результата ===
print("\nРезультат предсказания:")
if prediction == 1:
    print(f"⚠️ Высокая вероятность диабета ({prob*100:.1f}%)")
else:
    print(f"✅ Вероятность диабета низкая ({prob*100:.1f}%)")
