# from flask import Flask, render_template,request
# import numpy as np
# import pickle
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # from tensorflow.keras.models import load_model

# # model = load_model("model.h5")
# # from keras.metrics import mean_squared_error

# # # import pickle
# # from keras.models import load_model

# # custom_objects = {'mse': mean_squared_error}
# # # loaded_model = keras.saving.load_model("model.h5")

# # import keras.backend as K
# # from keras.utils.generic_utils import register_keras_serializable

# # @register_keras_serializable()
# # def custom_mse(y_true, y_pred):
# #     return K.mean(K.square(y_true - y_pred))

# # import tensorflow.keras.backend as K
# # from keras.utils.generic_utils import register_keras_serializable

# # @register_keras_serializable()
# # def custom_mse(y_true, y_pred):
# #     return K.mean(K.square(y_true - y_pred))

# from keras.models import load_model
# # import tensorflow.keras.backend as K
# import tensorflow.keras.backend as K

# # Define your custom loss function
# def custom_mse(y_true, y_pred):
#     return K.mean(K.square(y_true - y_pred))

# # Load the Keras model with custom objects
# try:
#     # model = load_model("model.h5", custom_objects={'custom_mse': custom_mse})\
#         model = load_model("model.h5", custom_objects={'custom_mse': custom_mse})

# except Exception as e:
#     print("Error loading model:", e)



# scaler = MinMaxScaler()

# # model=pickle.load(open("forecast_modell","rb"))

# app = Flask(__name__)

# @app.route('/',methods=['POST'])
# def index():
#     attr1 = float(request.form['maxTemp'])
#     attr2 = float(request.form['minTemp'])
#     attr3 = float(request.form['Humidity'])
#     attr4 = float(request.form['precipitation'])
#     attr5 = float(request.form['Pressure'])
#     attr6 = float(request.form['WindSpeed'])

#     input_data.append(attr1)
#     input_data.append(attr2)
#     input_data.append(attr3)
#     input_data.append(attr4)
#     input_data.append(attr5)
#     input_data.append(attr6)

#     input_data = np.array(input_data)
#     pred = model.predict(input_data)
#     pred2 = scaler.inverse_transform(pred)

    
#     return render_template('index.html',data=pred2)


# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request
# import numpy as np
# from keras.models import load_model

# app = Flask(__name__)

# # Load the Keras model
# model = load_model('model.h5')

# # Function to preprocess user inputs
# def preprocess_input(temp_max, temp_min, precipitation, humidity, pressure, wind_speed):
#     # Convert inputs to numpy array
#     inputs = np.array([[temp_max, temp_min, precipitation, humidity, pressure, wind_speed]])
#     return inputs

# # Function to make weather prediction
# def predict_weather(inputs):
#     # model.compile(optimizer='adam', loss='mean_squared_error')

#     prediction = model.predict(inputs)
#     return prediction[0][0]  # Assuming single output for weather prediction

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get user inputs from the form
#         temp_max = float(request.form['temp_max'])
#         temp_min = float(request.form['temp_min'])
#         precipitation = float(request.form['precipitation'])
#         humidity = float(request.form['humidity'])
#         pressure = float(request.form['pressure'])
#         wind_speed = float(request.form['wind_speed'])

#         # Preprocess inputs
#         inputs = preprocess_input(temp_max, temp_min, precipitation, humidity, pressure, wind_speed)

#         # Make prediction
#         prediction = predict_weather(inputs)

#         return render_template('result.html', prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.losses import MeanSquaredError

app = Flask(__name__)

# Load the Keras model
def load_weather_model():
    global model
    model = load_model('./model.h5')
    # Compile the model with the desired loss function
    # model.compile(optimizer='adam', loss='mean_squared_error')

load_weather_model()

# Function to preprocess user inputs
def preprocess_input(temp_max, temp_min, precipitation, humidity, pressure, wind_speed):
    # Convert inputs to numpy array
    inputs = np.array([[temp_max, temp_min, precipitation, humidity, pressure, wind_speed]])
    return inputs

# Function to make weather prediction
def predict_weather(inputs):
    model.compile(optimizer='adam', loss=MeanSquaredError())

    prediction = model.predict(inputs)
    return prediction[0][0]  # Assuming single output for weather prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        precipitation = float(request.form['precipitation'])
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])
        wind_speed = float(request.form['wind_speed'])

        # Preprocess inputs
        inputs = preprocess_input(temp_max, temp_min, precipitation, humidity, pressure, wind_speed)

        # Make prediction
        prediction = predict_weather(inputs)

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
