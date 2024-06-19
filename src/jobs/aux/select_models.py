#!/bin/python

#------------------- Description & Notes --------------------#

## To get an idea of how many sequences we can analyse:
##    262,144 total possible 9-mers
## == 131,072 Rev-Comp 9-mers
## =~ 1.04 MB

##    4,194,304 total possible 11-mers
## == 2,097,152 Rev-comp 11-mers
## =~ 16.77 MB

##    67,108,864 total possible 13-mers
## == 33,554,432 Rev-comp 13-mers
## =~ 268.43 MB

## This could be a solution to loading bigger datasets
## However, the issue with this will be how to incorporate k-fold cross-validation.
## https://stackoverflow.com/questions/60486804/dataset-doesnt-fit-in-memory-for-lstm-training

#------------------- Dependencies ---------------------------#

# Standard library imports
import time
import joblib
from pathlib import Path

# External imports
import numpy as np
import pandas as pd
from sklearn import model_selection

# Internal imports
from ...shared.model.common import *
from ...shared.kmer.common import *
from ...shared.common import *
from ...shared import io

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def runSelect(iDirs, clsFiles, oDir):
    print("Selecting model")

    ## Prepare data
    kmerDf = io.kmer.read(*iDirs)
    clsDf  = readClassInfo(clsFiles)
    kmerDf = prepareData(kmerDf, clsDf)
    (trainDf, testDf) = shuffleAndSplitData(kmerDf)

    ## Build, train and evaluate on training data
    clfs   = _buildClassifiers(trainDf)
    clfs   = [i[0] for i in clfs]
    trainScores = [_evaluateClassifier(clf, trainDf) for clf in clfs]
    trainScores = pd.concat(trainScores).reset_index(drop=True)
    trainScores = trainScores.round(3)

    ## Test and evaluate model on test data
    (train_x, train_y) = getInputOutputVariables(trainDf)
    clfs      = [clf.fit(train_x, train_y) for clf in clfs]
    testResults = [_getPredictions(clf, testDf) for clf in clfs]
    testResults = pd.concat(testResults).reset_index(drop=True)
    testResults = testResults.round(3)

    ## Write the tables to disk
    p = Path(oDir)
    p.mkdir(parents=True, exist_ok=True)

    f = '{}/trainScores.tsv'.format(oDir)
    trainScores.to_csv(f, sep='\t', index=False)

    f = '{}/testResults.tsv'.format(oDir)
    testResults.to_csv(f, sep='\t', index=False)

def runTrain(iDirs, clsFiles, oDir):
    print("Training model")

    ## Prepare data
    kmerDf = io.kmer.read(*iDirs)
    clsDf  = readClassInfo(clsFiles)
    kmerDf = prepareData(kmerDf, clsDf)

    ## Build and train the best model on all data
    clfs = _buildClassifiers(kmerDf)
    clfs = [i[0] for i in clfs]
    (kmerDf_x, kmerDf_y) = getInputOutputVariables(kmerDf)
    clfs = [_fitClassifier(clf, kmerDf_x, kmerDf_y) for clf in clfs]

    ## Write the tables to disk
    p = Path(oDir)
    p.mkdir(parents=True, exist_ok=True)

    ## Write the models to disk
    clfNames = [type(clf[-1]).__name__ for clf in clfs]
    clfNames = ['{}/{}.joblib'.format(oDir, n) for n in clfNames]
    [joblib.dump(clf, f) for clf, f in zip(clfs, clfNames)]

#------------------- Private Classes & Functions ------------#

def _buildClassifiers(trainDf):
    def _getClassifiers(trainDf):
        from sklearn.dummy import DummyClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.tree import ExtraTreeClassifier
        from sklearn.naive_bayes import GaussianNB

        iDim   = trainDf.iloc[:, 4:].shape[1]
        clfs = [
            (
                DummyClassifier(strategy='most_frequent'),
                dict()
            ),
            (
                KNeighborsClassifier(),
                dict(weights=['distance'],
                    n_neighbors=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
                )
            ),
            (
                MLPClassifier(max_iter=2000),
                dict(activation=['relu', 'tanh'],
                    batch_size=[64, 128, 256, 512],
                    hidden_layer_sizes=[(100, ), (100, 50, ), (int(iDim/2), )]
                )
            ),
            ## OK performing classifiers
            (
                LogisticRegression(max_iter=1500),
                dict(
                    C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
                )
            ),
            (
                RandomForestClassifier(),
                dict(
                    n_estimators=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                    max_features=['log2', 'sqrt', None]
                )
            ),
            ## Worst performing classifiers
            # (
            #     AdaBoostClassifier(),
            #     dict()
            # ),
            # (
            #     DecisionTreeClassifier(),
            #     dict()
            # ),
            # (
            #     ExtraTreeClassifier(),
            #     dict()
            # ),
            (
                GaussianNB(),
                dict()
            ),
        ]
        return clfs

    def _createPipeline(classifier):
        from sklearn.pipeline import Pipeline

        steps = [('classify', classifier[0])]
        pipe  = Pipeline(steps)
        pg    = {'classify__{}'.format(k): v for k,v in classifier[1].items()}
        return (pipe, pg)

    ## Build several models
    ## Machine learning (linear)     - Logistic regression
    ## Machine learning (non-linear) - Naive Bayes / Random forest / kNN
    ## Deep learning                 - Multi layer Perceptron (neural network; current implementation)
    clfs = _getClassifiers(trainDf)
    pipelines = [_createPipeline(clf) for clf in clfs]
    [print(i[0]) for i in pipelines]
    return pipelines

def _evaluateClassifier(clf, trainDf):
    ## Prepare the data
    (train_x, train_y) = getInputOutputVariables(trainDf)

    ## KFold cross validation
    scoring = [
        'accuracy', 'precision', 'recall',
        'f1', 'neg_mean_absolute_error'
    ]
    cv     = model_selection.RepeatedStratifiedKFold(n_repeats=5, n_splits=5)
    scores = model_selection.cross_validate(clf, train_x, train_y, cv=cv,
        return_train_score=True, scoring=scoring, n_jobs=-1)

    ## Summarise results
    scores['model'] = clf[-1].__str__()
    scores = pd.DataFrame(scores)
    scores = scores[['model'] + scores.columns.tolist()[:-1]]
    return scores

def _getPredictions(clf, testDf):
    ## Prepare the data
    testId = testDf.iloc[:, :4].copy()
    testCount = testDf.iloc[:, 4:]
    testCount.columns = testCount.columns.astype(str)

    ## Make predictions
    pred = clf.predict(testCount)
    prob = clf.predict_proba(testCount)
    prob = prob[:, 1]
    clsStr = np.where(pred == 1, 'Synthetic', 'Not synthetic')

    ## Keep track of the results
    testId['model']              = clf[-1].__str__()
    testId['predicted_classIdx'] = pred
    testId['predicted_class']    = clsStr
    testId['predicted_prob']     = prob
    return testId
    
def _fitClassifier(clf, kmerDf_x, kmerDf_y):
    start = time.time()
    clf   = clf.fit(kmerDf_x, kmerDf_y)
    end   = time.time()

    total = end - start
    total = str(round(total, 2))
    print('Execution of {} took {} seconds'.format(clf[-1].__str__(), total))
    return clf

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
