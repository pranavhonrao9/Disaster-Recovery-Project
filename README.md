# Disaster-Recovery-Project

In this data science Project, real messages  sent during disaster events are used to build machine learning model. This model  categorize events in proper manner so  messages can be sent to appropriate disaster relief agency and web application is created using flask.

Data : Figure Eight(https://www.figure-eight.com/)

# Motivation
In natural disasters , it is always challenging to provide right help at the right place in rigt time. So by creating whole end-end data science project , I was tryin to solve the issue and provide faster means of help.


# Files Description 

1. ETL Pipeline
In a Python script, process_data.py, data cleaning pipeline contains:

a.Loads the messages and categories datasets

b.Merges the two datasets

c.Cleans the data

d.Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py,a machine learning pipeline contains:

a.Loads data from the SQLite database

b.Splits the dataset into training and test sets

c.Builds a text processing and machine learning pipeline

d.Trains and tunes a model using GridSearchCV

e.Outputs results on the test set

f.Exports the final model as a pickle file

3. Web App
In run.py file ,web app task is written using flask and python.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
