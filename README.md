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

This version consists in preparing the database from data sheets for
 further modeling.
This version takes two data sheets as input to provide a database.
Development phase was performed on Jupyter Notebook and finally
 implemented into an etl_pipeline.py.


Prepare the database:

1. Import libraries and load datasets.
2. Merge datasets
3. Split categories into separate category columns
4. Convert category values to just numbers 0 or 1.
5. Replace categories column in df with new category columns.
6. Remove duplicates.
7. Save the clean dataset into an sqlite database.


Correction following review:

Change request:
"Please convert all categories to binary"
Answer:
use of "pd.get_dummies(...)" functions in the file "etl_pipeline.py".

Change request:
"Github & Code Quality - Doesn’t convert all categories to binary"
Answer:
I've replaced .astype(int) by pd.get_dummies(...) to get binary values into the categories.

  
VERSION:

ID: 1.0.1
This version is the initial version


PUBLIC RELEASE  

You can find the published released here:
https://github.com/laurent-jpc/UDACITY_PROJECT_DATA_ENGINEERING


INSTRUCTIONS:

Prepare the database:

- Create a workspace to implement the project.
- Create a folder "data", and load within it both csv data sheets 
  "disaster_categories.csv" and "disaster_messages.csv", and the
  python script "etl_pipeline.py".
- To launch the processing, select the directory of this folder 
  in a terminal (or equivalent) and,
- Enter the following command line 'python data/etl_pipeline.py 
  data/disaster_messages.csv data/disaster_categories.csv data/
  DisasterResponse.db'
- Check the processing has created a file "DisasterResponse.db"
  into the "data" folder.

ENVIRONNEMENT:

Python 3.6.3 packaged by conda-forge, run under the Jupyter 
 Notebook server in version 5.7.0.
For further details, refer to requirements.txt


REPOSITORY'S FILE:

- File "README.md"
- File "requirements.txt"
- File "etl_pipeline.py", python script to prepare the database
  of messages
- File "disaster_categories.csv", data sheet with categories
- File "disaster_messages.csv", data sheet with messages


DATA SOURCE:

Data are provided by https://learn.udacity.com/


ACKNOLEDGMENTS:

Data set credit: https://appen.com/


LEGAL / LICENSE:

Refer to Udacity and APPEN data access rights
