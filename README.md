# Disaster-Recovery-Project

In this data science Project, real messages  sent during disaster events is used to build machine learning model. This model  categorize events in proper manner so  messages can be sent to appropriate disaster relief agency and web application is created using flask.

Dataset used: disaster data from Figure Eight(https://www.figure-eight.com/)



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
