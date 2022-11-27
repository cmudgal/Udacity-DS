# Disaster Response Pipeline Project


### Table of Contents

1. [Installation](#installation)
2. [Project Objective](#objective)
3. [Project Pipelines Details](#details)
4. [Instructions](#instructions)

## Installation <a name="installation"></a>

Following libraries are required to be imported for pipeline and data processing:

- pandas
- re
- sys
- numpy
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

## Project Objective<a name="objective"></a>

The goal of the project is to classify the disaster messages into correct categories closest to the incident. 
In this project, disaster data from Appen is used to build a model using ML pipeline that analyzes and classifies disaster messages. 
A web app is built that allows a user to input a new message and get classification results in several categories. The web app also display visualizations of the data. This categorization results can then be sent to relief agencies for support.


## Project Detail<a name = "detail"></a>
The project has three sections which are:

1. **ETL Pipeline:** `process_data.py` file contain the script to create ETL pipline which:

- Loads csv format data from the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. **ML Pipeline:** `train_classifier.py` file contain the script to create ML pipline which:

- Loads data from the SQLite database
- Tokenize the data
- Splits the dataset into training and test sets
- Build a model using machine learning pipeline
- Train model using GridSearchCV
- Writes the final model as a pickle file

3. **Flask Web App:** the web app enables the user to enter a disaster message the output is to display categories of the message.

The web app also contains some visualizations that describe the data. 
 
 
 
## Files Descriptions <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- DisasterResponse
		- \app
			- run.py: flask file to run the app
		- \templates
			- master.html: main page of the web application 
			- go.html: result web page
		- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- DisasterResponse.db: disaster response database
			- process_data.py: ETL process
		- \models
			- train_classifier.py: classification code
                        - classifier.pk: pickle file
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
