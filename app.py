from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model/linear_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    price = model.predict([[area]]).item()
    return render_template(
        'index.html',
        prediction=f'Estimated Price ₹: {price:.2f}'
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)

