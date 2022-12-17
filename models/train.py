import sys
# import libraries
import re
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from sqlalchemy import create_engine

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    DESCRIPTION
        Loading data from database and split it into messages (X) and categories (Y)
    INPUT
        database_filepath is the file path of the database to process
    OUTPUT
        X is a numpy array of all messages only (strings)
        Y is a numpy array of all other categories relates to these messages (except 'original' messages -Fr, UK, US ...-
          in column #2, and except 'genre' in column #3
        category_names is names for Y categories
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='DisasterResponse', con=engine)  # load the Disaster Response table cleaned
    X = df["message"].values  # get column of messages
    Y = np.asarray(df[df.columns[4:]].values) # get categories columns
    Y = Y.astype(int)  # convert all Y values to int  # help from https://knowledge.udacity.com/questions/59448
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    '''
    DESCRIPTION
        text tokenization
    INPUT
        text is a list of messages (strings)
    OUTPUT
        clean_tokens is XXXX as result of the tokenization, which includes
         - normalizing the text, i.e. keep only spaces, letters and figures
         - Tokenization itself
         - Stop words (removal of all not english words; i.e. the X columns is already a conversion of
           messages from various languages (Fr, En ...) to english, thus no need to consider another language
         - Lemmatization
           including a change for Lower casing + remove leading and trailing spaces

    '''
    
    # Normalize, remove all that is not letters and figures
    pure_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    
    # split sentences into word
    tokens = word_tokenize(pure_text)
    
    # remove stop words from the sentences
    stop_words = stopwords.words('english')  # assume that there is only english sentences
    stop_word = [word for word in tokens if word not in stop_words]
    
    # normalize words keeping the source word according to the context
    #  and get only lower characters and remove leading and trailing characters
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for word in stop_word:
        lemmatized = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(lemmatized)    

    return clean_tokens


def build_model():
    '''
    DESCRIPTION
        Build a model through pipeline with RandomForestClassifier and GridSearchCV
    INPUT
        nil
    OUTPUT
        pipeline is the Pipeline defined for classification of data
    '''

    # Build the pipeline
    # - using MultiOutputClassifer link and refering to https://knowledge.udacity.com/questions/158218
    
    parameters = {
        'clf__estimator__n_estimators': [20]
    }

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),        
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs= -1)
    
    # MultiOutputClassifier (fit one classifier per target):
    #  n_estimators = 10
    #  n = -1  means using all available processes / threads

    return model


# --------- BEGINNING OF ADDITIONAL functions -------------
def store_value(values, index):
    try:
        value = values[index]
    except:
        value = np.nan
    return value


def store_float_value(values, index):
    try:
        value = float(values[index])
    except:
        value = np.nan
    return value



def get_scores_dict(report_txt, titles):
    '''
    DESCRIPTION
        Transform a text of report into a global dictionary
    INPUT
        report_txt is report under text format
        titles is the list of names for every results table
    OUTPUT
        dico is a dictionary of titles with related result per values 0, 1, total and related scores
        {title_1: {'0': {'precision': x, 'recall: x', 'f1-score': x, 'support': x}, '1': {...}, 'total': {...} }, title_2: ...}
    
    A line of the report contains:
       report A      precision  recall  f1-score  support
               0        x          x        x        x
               1        x          x        x        x
               total    x          x        x        x
    Sometimes, 0 or 1 my be not present
    '''
    
    lst = []  # initiate a list of list i.e. a list (lst) containing all words of every table contained in a related list (mots)
    
    # Read line by line of the report; one table result is contained on a line
    for tab in report_txt:
        # create a table with all elements, usung space as separator
        li = tab.split(' ')
        mots = []  # initiate a list of word and figure that will be extract from the line
        # Read the list of word read on the line
        for i, elt in enumerate(li):
            if elt != '':  # when word contains characters
                if i == 0:            # when this is the first word
                    mots.append(elt)  # - store this word in the list mots
                elif '\n' in elt:     # when the word contains a line break, indicating the end a of sub-line,
                                      #  it's time to change of sub-list
                    elt_f = elt.split('\n')[0]  # - get the first part of the word 
                    mots.append(elt_f)          # - and store it in the last list
                    lst.append(mots)            # - and store this list mtos into the list lst, before initiating a new list mots
                    elt_s = elt.split('\n')[1].replace('\n','')  # - get the second part of the word
                    if elt_s != '':             # - when the second of the word contains characters
                        mots = [elt_s]          # - store this new word into the new list mots
                    else:                       # - otherwise
                        mots = []               # - initiate a empty list mots
                else:
                    mots.append(elt)   # otherwise (not the first word, not the last word of a sub-line), then store the word
                                       #  into the list mots
        lst.append(mots)  # Finally, store every mots (containing words of every sub-line)
    
    # Now that we have list and sub-list of words of the report
    #  we gonna get values per label
    dico = dict()
    start_cycle = -1
    di = dict()
    k = -1
    # Read every sub-list of words of the list
    for i, t in enumerate(lst):
        label = ''        
        values = t[-4:]  # Get last four values of every line
        if len(values) > 0:  # if there are values
            if values[0] == 'precision': # detect the word 'precision' as starter
                k += 1  # iterate k as index to join title and related data
                start_cycle = i  # store i
                if len(di) > 0:          # if we fulfill di (dict) during previous lines
                    dico[titles[k]] = di #  then we store it in dico 
                di = dict()              # any way, at thi stage of the starter, we initiate dict di
                                         # di is a dict to store every result values 
            else:  # when we are in another line that the starter
                if i > start_cycle: 
                    label = label.join(t[:(len(t)-4)]).replace('/', '') # we get the label of this line
                    # and then we get score values for every category of result, storing that in a dictionary d
                    d = dict()  # d for storing scores of a label
                    d['precision'] = store_value(values, 0)
                    d['recall'] = store_value(values, 1)
                    d['f1-score'] = store_value(values, 2)
                    d['support'] = store_value(values, 3)
                    di[label] = d  # when we get all scores, we store it into dictionary di, di = scores for all labels of a table

    return dico


def get_feature_values(dico, feature):
    '''
    DESCRIPTION
        Reading a dictionary of result (from get_scores), we focus on a feature of results to get these values
    INPUT
        dico is a dictionary of results, multi tables, multi-lines with result features for every line
    OUTPUT
        dico_feat is a dictionary of the result-score for the selected result-feature
        {title_1: [0_score, 1_score, total_score], ...}, nan is no value
    '''

    dico_feat = dict()
    # Read every table of result
    for i, j in dico.items():
        k0, k1, k2 = np.nan, np.nan, np.nan  # initiate scores for the selected result-feature
        
        # Read every label and relates scores of a table
        for k, m in j.items():
            #  get score for every label
            if k == '0':
                k0 = store_float_value(m, feature)
            elif k == '1':
                k1 = store_float_value(m, feature)
            else:
                k2 = store_float_value(m, feature)
        # store scores into a dictionary
        dico_feat[i] = [k0, k1, k2]
    return dico_feat


def get_list_value(lst, pos):
    try:
        r = lst[pos]
    except:
        r = np.nan
    return r


def dict_to_df(dico):
    '''
    DESCRIPTION
        transform a dictionary of scores (from get_feature_values) into a dataframe
    INPUT
        dico is a dictionary of the score per title
              under format {title_1: [0_score, 1_score, total_score], ...}, nan is no value
    OUTPUT
        df is a dataframe with these scores, under format
        columns = items, '0', '1', 'total' ; rows = titles ; values = score-values
    '''

    col = ['items', '0', '1', 'total']
    items, zero, un, total = [], [], [], []
    for i, j in dico.items():
        items.append(i)
        # It's possible not to have a value for 0, 1 or total, so we have to manage it by proposing an alternate solution
        zero.append(get_list_value(j, 0))
        un.append(get_list_value(j, 1))
        total.append(get_list_value(j, 2))

    d = {'items': items, '0': zero, '1': un, 'total':total}
    df = pd.DataFrame(data=d)
    df[['0', '1', 'total']].astype(float)  # change type values to float
    return df


def get_f1_score(report, titles, result_feature='f1-score'):
    '''
    DESCRIPTION
        Transform a report of scoresinto a dataframe of the selected result-feature (here: f1-score)
    INPUT
        report is the report of scores (string)
        titles is the list of names of all parameters
        result_feature is the name of the result-feature that we want to get ; f1-score per default
    OUTPUT
        df_feature is a dataframe with result-scores of the selected result-feature for all labels and parameters
    '''
    dico = get_scores_dict(report, titles)  # Transform a text of report into a global dictionary
    dico_feature = get_feature_values(dico, result_feature)  # Reading a dictionary of result (from get_scores),
                                                             #  we focus on a feature of results to get these values
    df_feature = dict_to_df(dico_feature)  # transform a dictionary of scores (from get_feature_values) into a dataframe
    
    return df_feature
# --------- END OF ADDITIONAL functions -------------


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    DESCRIPTION
        evaluate the model on test data
    INPUT
        model is the trained model to evaluate
        X_test is the test input data set 
        Y_test is the test output data set
        category_names is the list of names of all categories
    OUTPUT
        nil
    '''    
    Y_pred = model.predict(X_test)
    
    # F1 score with MultiLabelBinarizer and f1_score
    mlb = MultiLabelBinarizer().fit(Y_test)
    f1_score_mlb = f1_score(mlb.transform(Y_test),
         mlb.transform(Y_pred),
         average='macro')
    print(' -> f1 score mlb:', f1_score_mlb)
        
    # F1 score with a classification report
    # Then using the classificatio_report mentioned for the exercice:
    # refering to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    scores_class_report = []
    # Run over all categories to get their own individual score
    for i in range(len(category_names)):
        # get a report, i.e. a text, with classification results, including scores for 0, 1 and total
        report = classification_report(np.array(Y_test)[i], np.array(Y_pred)[i], zero_division=1)
        # Warning may appear: it may be avoid with zero_division=0 & output_dict=True
        #  but they are available only with New in version 0.20. while we work on version 0.19!
        #  Refer to the import of Warnings and related processing
        #                     report_score = fscore(y_test, y_pred, average='weighted')  # TEST !
        #                     print('report score:\n', report_score)  # TEST !
        scores_class_report.append(report)  # We store all reports for a further processing
    # Now we have reports of classification for every categories
    #  we gonna get only score values, for 0, 1 and total for every category and resume that in a dataframe 
    df_scores_class_report = get_f1_score(scores_class_report, category_names)
    # then I can compute one score for all categories by computing the 'total' score over all categories.
    f1_score_cr = df_scores_class_report["total"].mean(axis=0)
    print(' -> f1 score cr:', f1_score_cr)  
    

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
                
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()