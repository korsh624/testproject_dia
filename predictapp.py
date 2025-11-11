import joblib
import numpy as np
model=joblib.load('diabetes_model.pkl')
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
print("Введите данные по каждому признаку:")
user_data=[]
for feature in feature_names:
    while True:
        try:
            value=float(input(f'{feature}: '))
            user_data.append(value)
            break
        except ValueError:
            print('Input numbers ')
user_array=np.array(user_data).reshape(1,-1)
user_scaled=scaler.transform(user_array)
prediction=model.predict(user_scaled)[0]
if prediction==1:
    print('Hig value diabet')
else:
    print('low value diabet')