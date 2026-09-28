import os
import pickle
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'random_forrest_titanic.pkl')
model = None

try:
  with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
  print('Successfully loaded random_forrest_titanic.pkl')
except Exception as e:
  print(f'Warning: Could not load model: {e}')



@app.route('/')
def home():
  # Serving the new 2D Map HTML
  return render_template('map.html')


# Route to serve animation frames from the 'map-animation' folder
@app.route('/map-animation/<path:filename>')
def serve_map_animation(filename):
  anim_dir = os.path.join(os.path.dirname(__file__), 'map-animation')
  return send_from_directory(anim_dir, filename)

# Add this right below your other routes in app.py
@app.route('/Character Sequence/<path:filename>')
def serve_character_sequence(filename):
    char_dir = os.path.join(os.path.dirname(__file__), 'Character Sequence')
    return send_from_directory(char_dir, filename)


@app.route('/Portal-Blast-1/<path:filename>')
def serve_portal(filename):
  portal_dir = os.path.join(os.path.dirname(__file__), 'Portal-Blast-1')
  return send_from_directory(portal_dir, filename)

@app.route('/cinematics/<path:filename>')
def serve_cinematics(filename):
    cinematics_dir = os.path.join(os.path.dirname(__file__), 'cinematics')
    return send_from_directory(cinematics_dir, filename)

@app.route('/TitleCard/<path:filename>')
def serve_title_card(filename):
    title_dir = os.path.join(os.path.dirname(__file__), 'TitleCard')
    return send_from_directory(title_dir, filename)

# Machine Learning Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json() or {}

  pclass = int(data.get('pclass', 3))
  sex = int(data.get('sex', 1))
  age = float(data.get('age', 28.0))
  sibsp = int(data.get('sibsp', 0))
  parch = int(data.get('parch', 0))
  fare = float(data.get('fare', 32.0))
  embarked = str(data.get('embarked', 'S')).upper()

  sex_male = 1 if sex == 1 else 0
  embarked_q = 1 if embarked == 'Q' else 0
  embarked_s = 1 if embarked == 'S' else 0

  input_df = pd.DataFrame([{
      'PassengerId': 1310,
      'Pclass': pclass,
      'Age': age,
      'SibSp': sibsp,
      'Parch': parch,
      'Fare': fare,
      'Sex_male': sex_male,
      'Embarked_Q': embarked_q,
      'Embarked_S': embarked_s,
  }])

  if model is not None:
    try:
      pred = int(model.predict(input_df)[0])
      prob = float(model.predict_proba(input_df)[0][1]) * 100
    except Exception as e:
      print(f'Prediction error: {e}')
      prob = 85.0 if sex == 0 and pclass <= 2 else 18.0
      pred = 1 if prob >= 50.0 else 0
  else:
    prob = 80.0 if (sex == 0 or age < 12) and pclass <= 2 else 15.0
    pred = 1 if prob >= 50.0 else 0

  return jsonify({
      'survived': bool(pred == 1),
      'probability': round(prob, 1),
  })


if __name__ == '__main__':
  app.run(debug=True, port=5000,threaded=True)