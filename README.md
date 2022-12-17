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

After build of the database, processed the database and export
 a trained model, this version consists in providing all missing
 component of the API: It takes a database and a classifier model as inputs to
 interact with the web app's user interface.


Prepare the database:

1. Import libraries and load datasets.
2. Merge datasets
3. Split categories into separate category columns
4. Convert category values to just numbers 0 or 1.
5. Replace categories column in df with new category columns.
6. Remove duplicates.
7. Save the clean dataset into an sqlite database.

Process the database and export a trained model:

1. Import libraries and load data from database.
2. Tokenize the text data
3. Build a machine learning pipeline
4. Train pipeline
5. Test the model #1
6. Improve the model #1
7. Test the improved model #2
8. Improve the model #2
9. Export the model as a pickle file

Provide the elements of the API:
 - Provide an interface to enter a new message and classify it
   by selecting the related categories in the list of categories
   below the input field
 - Provide vizualizations related to the database
 

VERSION:

ID: 1.0.4
This version add the requested folders in the repository.


PUBLIC RELEASE  

You can find the published released here:
https://github.com/laurent-jpc/UDACITY_PROJECT_DATA_ENGINEERING


INSTRUCTIONS:

Prepare the database:

- Create a workspace to implement the project.
- Ensure that both csv data sheets "disaster_categories.csv" and
  "disaster_messages.csv", and the python script "etl_pipeline.py"
  are available in the folder "data".
- To launch the processing, select the directory of this folder
  in a terminal (or equivalent) and,
- Enter the following command line 'python data/etl_pipeline.py data/
  disaster_messages.csv data/disaster_categories.csv data/
  DisasterResponse.db`
- Check the processing has created a file "DisasterResponse.db"
  into the "data" folder.

Process the database and export a trained model:

- Ensure the DisasterResponse.db file, build from the first phase is
  available into the "data" folder.
- Ensure the file "train.py" file is available in the folder "models".
- To launch the processing, select the directory of this folder in
  a terminal (or equivalent) and,
- Enter the following command line 'python models/train.py data/
  DisasterResponse.db models/classifier.pkl`
- Check the processing has created a file "classifier.pkl" into the
  "models" folder.

Run the API: (NEW)

- Ensure files "DisasterResponse.db" and classifier.pkl" built at
  previous steps are available in their appropriate folder.
- Ensure the file "run.py" is available in the folder "app".
- Ensure that files "go.html" and "master.html" are available in 
  the folder "templates" into the folder "app".
- Go to 'app' directory by entering 'cd app'
- Then run the web app by entering 'python run.py'


ENVIRONNEMENT:

Python 3.6.3 packaged by conda-forge, run under the Jupyter Notebook
 server in version 5.7.0.
For further details, refer to requirements.txt


REPOSITORY'S FILE:

- File "README.md"
- File "requirements.txt"
- Folder "data" containing:
  > File "etl_pipeline.py", python script to prepare the database
  of messages;
  > File "disaster_categories.csv", data sheet with categories;
  > File "disaster_messages.csv", data sheet with messages;
  > File "DisasterResponse.db", database with messages and categories
    built by running the file "etl_pipeline.py".
- Folder "models" containing:
  > File "train.py", python script to process the database and export
    a trained model
  > File "classifier.pkl" contains the trained model to classify new
    messages. it it built byt running the file "train.py".
- Folder "app" containing:
  > File "run.py" is a python script to make the interface interactive,
    applying the model on the new message entered by the emergency user
    and providing content for displaying vizualization related to content
    of the database.
  > Folder "templates" containing:
	Files "go.html" and "master.html" to get the design and the
    information that we need to run the web app.


DATA SOURCE:

Data are provided by https://learn.udacity.com/


ACKNOLEDGMENTS:

Data set credit: https://appen.com/


LEGAL / LICENSE:

Refer to Udacity and APPEN data access rights
