TITLE:

UDACITY PROJECT DATA ENGINEERING "Disaster Response Pipelines"


PROJECT:

Purpose of the project consists propose a web app to emergency worker 
 that help classify instantaeously new disaster messages in several
 categories.
With such a classification, the emergency worker can send the messages
 to the appropriate disaster relief agency.
The web app is based on model trained on the analysis of older disaster
 data provided from Appen.
An API will allow the emergency worker to interface with the web app by
 inserting new messsage, get the related categories and get some
 vizualizations of the database content.


PROJECT MOTIVATION:

As part of my Udacity Data scientist training, this project allows
 implementing a machine learning script.



DESCRIPTION:

After build of the database, this version consists in processing the
 messages to build and train a model through ETL pipeline.
This version takes a database as input and provide a model as output
 pickle file.
Development phase was performed on Jupyter Notebook and finally
 implemented into an train.py.


Prepare the database:

1. Import libraries and load datasets.
2. Merge datasets
3. Split categories into separate category columns
4. Convert category values to just numbers 0 or 1.
5. Replace categories column in df with new category columns.
6. Remove duplicates.
7. Save the clean dataset into an sqlite database.

Process the database and export a trained model: (NEW)

1. Import libraries and load data from database.
2. Tokenize the text data
3. Build a machine learning pipeline
4. Train pipeline
5. Test the model #1
6. Improve the model #1
7. Test the improved model #2
8. Improve the model #2
9. Export the model as a pickle file

Correction following review:

Change request:
"Please resolve the error in train_classifier.py script"
"Machine Learning - script runs with errors in the terminal"
Answer:
Actually, I tested the script on the Jupyter Notebook of Udacity with no error.
Neverthless in train.py there was a warning "UndefinedMetricWarning: F-score is ill-defined
 and being set to 0.0 in labels with no predicted samples" linked to "classification_report",
 the warning was cancelled by getting the last version of sklearn and adding the option
 "zero_division=1" (scores are slightly better with  than 0) as mentioned at
 "https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-
  and-being-set-to-0-0-in-labels-wi"
  

VERSION:

ID: 1.0.2
This version add the ML pipeline


PUBLIC RELEASE  

You can find the published released here:
https://github.com/laurent-jpc/UDACITY_PROJECT_DATA_ENGINEERING


INSTRUCTIONS:

Prepare the database:

- Create a workspace to implement the project.
- Create a folder "data" and load within it both csv data sheets
  "disaster_categories.csv" and "disaster_messages.csv", and the python
  script "etl_pipeline.py".
- To launch the processing, select the directory of this folder in a
  terminal (or equivalent) and,
- Enter the following command line 'python data/etl_pipeline.py data/
  disaster_messages.csv data/disaster_categories.csv data/
  DisasterResponse.db`
- Check the processing has created a file "DisasterResponse.db" into
  the "data" folder.

Process the database and export a trained model: (NEW)

- Ensure the DisasterResponse.db file, build from the first phase is
  available into the "data" folder.
- Create a folder "models" into the workspace, aside the folder "data"
  and load the "train.py" file within.
- To launch the processing, select the directory of this folder in a
  terminal (or equivalent) and,
- Enter the following command line 'python models/train.py data/
  DisasterResponse.db models/classifier.pkl`
- Check the processing has created a file "classifier.pkl" into the
  "models" folder.


ENVIRONNEMENT:

Python 3.6.3 packaged by conda-forge, run under the Jupyter Notebook
 server in version 5.7.0.
For further details, refer to requirements.txt


REPOSITORY'S FILE:

- File "README.md"
- File "requirements.txt"
- File "train.py", python script to process the database and export
  a trained model
- File "DisasterResponse.db", database with messages and categories
  prepare by the first part of the project


DATA SOURCE:

Data are provided by https://learn.udacity.com/


ACKNOLEDGMENTS:

Data set credit: https://appen.com/


LEGAL / LICENSE:

Refer to Udacity and APPEN data access rights
