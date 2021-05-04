from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('data.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods = ['POST'])
def result():
    data1 = request.form['a']
    data2 = request.form['b']
    arr = np.array([[data1 , data2]])
    prediction = model.predict(arr)
    return render_template('result.html', data = prediction)
if __name__ == '__main__':
    app.run(debug=True)