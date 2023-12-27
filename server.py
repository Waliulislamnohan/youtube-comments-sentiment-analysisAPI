from flask import Flask, render_template, request, jsonify
import joblib
import pickle

app = Flask(__name__)


model = joblib.load('svc_model.sav')


with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']


        text_vectorized = vectorizer.transform([text]).toarray()

        
        prediction = model.predict(text_vectorized)[0]

        
        sentiment = "Positive" if prediction == 1 else "Negative"

        return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)