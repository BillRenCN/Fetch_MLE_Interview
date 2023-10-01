# app.py
from flask import Flask, render_template, request

import main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        predict_days = int(request.form['look_ahead_days'])
        update = request.form.get('update_model') == 'on'

        # Call the main function and capture the monthly_predict result
        monthly_predict = main.main(update, predict_days, True)

        # Pass monthly_predict to the template
        return render_template('result.html', monthly_predict=monthly_predict)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
