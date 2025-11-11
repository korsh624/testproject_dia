import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackContext
import joblib
import numpy as np

# === Загрузка модели и кодировщиков ===
model = joblib.load("diabetes_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# === Логирование ===
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# === Состояния ===
(GENDER, AGE, HYPERTENSION, HEART, SMOKE, BMI, HBA1C, GLUCOSE) = range(8)

user_data = {}

def start(update: Update, context: CallbackContext):
    update.message.reply_text("👋 Привет! Я помогу оценить риск диабета.\nВведите ваш пол (Male / Female / Other):")
    return GENDER

def gender(update: Update, context: CallbackContext):
    user_data["gender"] = update.message.text.strip()
    update.message.reply_text("Введите возраст:")
    return AGE

def age(update: Update, context: CallbackContext):
    user_data["age"] = float(update.message.text)
    update.message.reply_text("Гипертония? (0 - нет, 1 - есть):")
    return HYPERTENSION

def hypertension(update: Update, context: CallbackContext):
    user_data["hypertension"] = int(update.message.text)
    update.message.reply_text("Болезни сердца? (0 - нет, 1 - есть):")
    return HEART

def heart(update: Update, context: CallbackContext):
    user_data["heart_disease"] = int(update.message.text)
    update.message.reply_text("Курение (never, current, former, No Info, ever, not current):")
    return SMOKE

def smoke(update: Update, context: CallbackContext):
    user_data["smoking_history"] = update.message.text.strip()
    update.message.reply_text("Введите BMI (индекс массы тела):")
    return BMI

def bmi(update: Update, context: CallbackContext):
    user_data["bmi"] = float(update.message.text)
    update.message.reply_text("Введите HbA1c уровень (например 5.8):")
    return HBA1C

def hba1c(update: Update, context: CallbackContext):
    user_data["HbA1c_level"] = float(update.message.text)
    update.message.reply_text("Введите уровень глюкозы в крови:")
    return GLUCOSE

def glucose(update: Update, context: CallbackContext):
    user_data["blood_glucose_level"] = float(update.message.text)

    # Кодируем признаки
    gender = label_encoders['gender'].transform([user_data['gender']])[0]
    smoke = label_encoders['smoking_history'].transform([user_data['smoking_history']])[0]

    X = np.array([[gender,
                   user_data['age'],
                   user_data['hypertension'],
                   user_data['heart_disease'],
                   smoke,
                   user_data['bmi'],
                   user_data['HbA1c_level'],
                   user_data['blood_glucose_level']]])

    # Предсказание
    prediction = model.predict(X)[0]
    result = "⚠️ Есть риск диабета. Рекомендуется обратиться к врачу." if prediction == 1 else "✅ Риск диабета не обнаружен."

    update.message.reply_text(result)
    update.message.reply_text("Чтобы начать заново — введите /start")
    return ConversationHandler.END

def cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Диалог завершён. Введите /start, чтобы начать снова.")
    return ConversationHandler.END

def main():
    # 🔑 ВСТАВЬ сюда свой токен от BotFather
    TOKEN = "8397219511:AAFNdRZ_JY5ypYMck1zF5SJQDcZEqVBtxSg"

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
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
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dp.add_handler(conv_handler)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
