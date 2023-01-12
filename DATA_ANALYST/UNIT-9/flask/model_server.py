import numpy as np
from flask import Flask, request

app = Flask(__name__)

def model_predict(value):
	return value ** 3

@app.route('/predict')
def predict_func():
	value = request.args.get('value')
	prediction = model_predict(int(value))
	return f'the result is {prediction}'

if __name__ == '__main__':
	app.run('localhost', 5000)