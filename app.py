from flask import Flask, request, jsonify
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model_path = "svm_gender_prediction_model.pkl"
loaded_model = joblib.load(model_path)

def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Save the uploaded audio file
        file = request.files['file']
        file_path = "uploaded_audio.wav"
        file.save(file_path)

        # Extract MFCC features
        mfcc_features = extract_mfcc(file_path).reshape(1, -1)

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(mfcc_features)

        # Make prediction
        prediction = loaded_model.predict(scaled_features)[0]
        gender = 'male' if prediction == 0 else 'female'

        return jsonify({'gender': gender})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
