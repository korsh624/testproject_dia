import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackContext
import joblib
import numpy as np

# === Загрузка модели и кодировщиков ===
model = joblib.load("diabetes_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# === Логирование ===
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# === Этапы разговора ===
(GENDER, AGE, HYPERTENSION, HEART, SMOKE, BMI, HBA1C, GLUCOSE) = range(8)
user_data = {}

# === Начало ===
def start(update: Update, context: CallbackContext):
    keyboard = [["Мужской", "Женский"]]
    update.message.reply_text(
        "👋 Привет! Я помогу оценить риск диабета.\n\nВыберите ваш пол:",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return GENDER


# === 1. Пол ===
def gender(update: Update, context: CallbackContext):
    text = update.message.text
    mapping = {"Мужской": "Male", "Женский": "Female", "Другое": "Other"}
    user_data["gender"] = mapping.get(text, "Other")

    update.message.reply_text("Введите ваш возраст (в годах):", reply_markup=ReplyKeyboardRemove())
    return AGE


# === 2. Возраст ===
def age(update: Update, context: CallbackContext):
    user_data["age"] = float(update.message.text)

    keyboard = [["Нет", "Да"]]
    update.message.reply_text(
        "Есть ли у вас гипертония?", reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    )
    return HYPERTENSION


# === 3. Гипертония ===
def hypertension(update: Update, context: CallbackContext):
    user_data["hypertension"] = 1 if update.message.text == "Да" else 0

    keyboard = [["Нет", "Да"]]
    update.message.reply_text(
        "Есть ли у вас заболевания сердца?",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return HEART


# === 4. Болезни сердца ===
def heart(update: Update, context: CallbackContext):
    user_data["heart_disease"] = 1 if update.message.text == "Да" else 0

    keyboard = [["Никогда", "Сейчас", "Раньше", "Нет данных", "Иногда", "Бросил"]]
    update.message.reply_text(
        "Ваше отношение к курению:",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return SMOKE


# === 5. Курение ===
def smoke(update: Update, context: CallbackContext):
    mapping = {
        "Никогда": "never",
        "Сейчас": "current",
        "Раньше": "former",
        "Нет данных": "No Info",
        "Иногда": "ever",
        "Бросил": "not current",
    }
    user_data["smoking_history"] = mapping.get(update.message.text, "No Info")

    update.message.reply_text("Введите ваш индекс массы тела (вес (кг) ÷ рост² (м) :", reply_markup=ReplyKeyboardRemove())
    return BMI


# === 6. BMI ===
def bmi(update: Update, context: CallbackContext):
    user_data["bmi"] = float(update.message.text)
    update.message.reply_text("Введите уровень HbA1c (например 5.8):")
    return HBA1C


# === 7. HbA1c ===
def hba1c(update: Update, context: CallbackContext):
    user_data["HbA1c_level"] = float(update.message.text)
    update.message.reply_text("Введите уровень глюкозы в крови (в мг/дл):")
    return GLUCOSE


# === 8. Глюкоза и предсказание ===
def glucose(update: Update, context: CallbackContext):
    user_data["blood_glucose_level"] = float(update.message.text)

    # Кодируем категориальные признаки
    gender = label_encoders["gender"].transform([user_data["gender"]])[0]
    smoke = label_encoders["smoking_history"].transform([user_data["smoking_history"]])[0]

    # Формируем входные данные
    X = np.array([[gender,
                   user_data["age"],
                   user_data["hypertension"],
                   user_data["heart_disease"],
                   smoke,
                   user_data["bmi"],
                   user_data["HbA1c_level"],
                   user_data["blood_glucose_level"]]])

    prediction = model.predict(X)[0]
    result = (
        "⚠️ Есть риск диабета. Рекомендуется обратиться к врачу."
        if prediction == 1
        else "✅ Риск диабета не обнаружен."
    )

    update.message.reply_text(result, reply_markup=ReplyKeyboardRemove())
    update.message.reply_text("Чтобы начать заново, введите /start")
    return ConversationHandler.END


# === Отмена ===
def cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Диалог завершён. Введите /start, чтобы начать заново.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


# === Основная функция ===
def main():
    # ⚠️ Вставь сюда токен от BotFather
    TOKEN = ""

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            GENDER: [MessageHandler(Filters.text & ~Filters.command, gender)],
            AGE: [MessageHandler(Filters.text & ~Filters.command, age)],
            HYPERTENSION: [MessageHandler(Filters.text & ~Filters.command, hypertension)],
            HEART: [MessageHandler(Filters.text & ~Filters.command, heart)],
            SMOKE: [MessageHandler(Filters.text & ~Filters.command, smoke)],
            BMI: [MessageHandler(Filters.text & ~Filters.command, bmi)],
            HBA1C: [MessageHandler(Filters.text & ~Filters.command, hba1c)],
            GLUCOSE: [MessageHandler(Filters.text & ~Filters.command, glucose)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    dp.add_handler(conv_handler)

    print("🤖 Бот запущен. Нажми Ctrl+C для остановки.")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
