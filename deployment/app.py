#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import joblib

# Buka vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan vectorizer
# with open('model/vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)

# with open('model/model.pkl', 'rb') as f:
#     model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_text = request.form['text']
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

