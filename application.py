from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create the Flask application instance.
# __name__ tells Flask where to look for templates and static files.
application = Flask(__name__)

# AWS Elastic Beanstalk expects the WSGI callable to be named 'application',
# but 'app' is the conventional Flask name — this alias lets both work.
app = application


@app.route('/')
def index():
    """Render the landing page of the web application.

    Returns:
        Response: Renders index.html — the home/welcome page.
    """
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Handle student score prediction requests.

    GET  — Display the blank input form (home.html) for the user to fill in.
    POST — Read the submitted form data, run the ML prediction pipeline,
           and re-render the form with the predicted math score.

    Form fields expected (POST):
        gender                      : Student's gender (e.g. 'male', 'female')
        ethnicity                   : Race/ethnicity group (e.g. 'group A')
        parental_level_of_education : Highest education level of parent
        lunch                       : Lunch type ('standard' or 'free/reduced')
        test_preparation_course     : Whether course was completed or not
        writing_score               : Student's writing score (mapped to reading_score field)
        reading_score               : Student's reading score (mapped to writing_score field)

    Returns:
        Response: Renders home.html.
                  On POST, passes `results` (predicted math score) to the template.
    """
    if request.method == 'GET':
        # Just show the empty prediction form
        return render_template('home.html')
    else:
        # --- Collect & wrap form inputs ---
        # CustomData bundles the raw form values into a structured object
        # and validates/converts types before they reach the model.
        # Note: reading_score and writing_score form fields are intentionally
        # swapped here (writing_score → reading_score and vice versa).
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),   # see note above
            writing_score=float(request.form.get('reading_score')),   # see note above
        )

        # Convert the structured input into a single-row pandas DataFrame
        # that matches the feature schema the model was trained on
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # --- Run the prediction pipeline ---
        # PredictPipeline loads the saved preprocessor and model artifacts,
        # applies the same transformations used during training, then predicts.
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)  # returns an array of predictions
        print("after Prediction")

        # results[0] extracts the single predicted value from the array
        # and passes it to the template as the 'results' context variable
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    # host="0.0.0.0" makes the server reachable on all network interfaces,
    # which is required when running inside a container or on a remote VM.
    app.run(host="0.0.0.0")
