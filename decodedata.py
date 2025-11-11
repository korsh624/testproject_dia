import pandas as pd

# === 1. Чтение исходного CSV ===
file_path = "diabetes_prediction_dataset.csv"  # замените путь, если файл в другой папке
df = pd.read_csv(file_path)

# === 2. Перевод заголовков ===
translations = {
    "gender": "gender (пол)",
    "age": "age (возраст)",
    "hypertension": "hypertension (гипертония)",
    "heart_disease": "heart_disease (болезни сердца)",
    "smoking_history": "smoking_history (история курения)",
    "bmi": "bmi (индекс массы тела)",
    "HbA1c_level": "HbA1c_level (уровень HbA1c)",
    "blood_glucose_level": "blood_glucose_level (уровень глюкозы в крови)",
    "diabetes": "diabetes (наличие диабета)"
}

# === 3. Переименование колонок ===
df = df.rename(columns=translations)

# === 4. Сохранение в Excel ===
output_path = "diabetes_prediction_translated.xlsx"
df.to_excel(output_path, index=False)

print(f"Файл успешно сохранён как {output_path}")
