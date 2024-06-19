#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import joblib
from pathlib import Path

# External imports
import numpy as np
import pandas as pd

from tensorflow.keras import models
from tensorflow.keras import callbacks

# Internal imports
from ...shared.model.common import *
from ...shared.kmer.common import *
from ...shared.common import *
from ...shared import io

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def runEvaluate(iDirs, clsFiles, modelFile, epochs, batchsize, oDir):
    print("Evaluating model")

    ## Load params
    p = {'batch_size':batchsize, 'epochs':epochs}
    emb_path = modelFile + '/SkClassifier'
    clf_path = modelFile + '/KerasClassifier'
    print(p)

    ## Prepare data
    kmerDf = io.kmer.read(*iDirs)
    clsDf  = readClassInfo(clsFiles)
    kmerDf = prepareData(kmerDf, clsDf)
    (trainDf, testDf) = shuffleAndSplitData(kmerDf)
    (trainDf_x, trainDf_y) = getInputOutputVariables(trainDf)

    ## Include embedding if we have a PCA layer
    if (Path(emb_path).exists()):
        ## Convert frequencies to embeddings
        emb = joblib.load(emb_path)
        trainDf_x = emb.fit_transform(trainDf_x)
        trainDf_x = pd.DataFrame(trainDf_x)

        testDf_x = emb.transform(testDf.iloc[:, 4:])
        testDf_x = pd.DataFrame(testDf_x)
        testDf = pd.concat([testDf.iloc[:, :4], testDf_x], axis=1)

    ## Training history
    (trainHistory, clf) = _getTrainingHistory(clf_path, p, trainDf_x, trainDf_y)

    ## Test and evaluate model on test data
    (testResults, testScores) = _evaluateClassifier(clf, testDf)

    ## Write the model to disk
    p = Path(oDir)
    p.mkdir(parents=True, exist_ok=True)

    f = '{}/eval_trainHistory.tsv'.format(oDir)
    trainHistory.to_csv(f, sep='\t', index=False)

    f = '{}/eval_testResults.tsv'.format(oDir)
    testResults.to_csv(f, sep='\t', index=False)

    f = '{}/eval_testScores.tsv'.format(oDir)
    testScores.to_csv(f, sep='\t', index=False)

def runTrain(iDirs, clsFiles, modelFile, epochs, batchsize, oDir):
    print("Training model")

    ## Load params
    p = {'batch_size':batchsize, 'epochs':epochs}
    emb_path = modelFile + '/SkClassifier'
    clf_path = modelFile + '/KerasClassifier'
    print(p)

    ## Prepare data
    kmerDf = io.kmer.read(*iDirs)
    clsDf  = readClassInfo(clsFiles)
    kmerDf = prepareData(kmerDf, clsDf)
    (kmerDf_x, kmerDf_y) = getInputOutputVariables(kmerDf)

    ## Include embedding if we have a PCA layer
    if (Path(emb_path).exists()):
        ## Convert frequencies to embeddings
        emb = joblib.load(emb_path)
        kmerDf_x = emb.fit_transform(kmerDf_x)
        kmerDf_x = pd.DataFrame(kmerDf_x)

    ## Training history
    (trainHistory, clf) = _getTrainingHistory(clf_path, p, kmerDf_x, kmerDf_y)

    ## Write the model to disk
    p = Path(oDir)
    p.mkdir(parents=True, exist_ok=True)

    f = '{}/trainHistory.tsv'.format(oDir)
    trainHistory.to_csv(f, sep='\t', index=False)

    f = '{}/{}'.format(oDir, 'KerasClassifier')
    models.save_model(clf, f)

    ## Write embedding if we have a PCA layer
    if (Path(emb_path).exists()):
        f = '{}/{}'.format(oDir, 'SkClassifier')
        joblib.dump(emb, f)

#------------------- Private Classes & Functions ------------#

def _getTrainingHistory(f, p, trainDf_x, trainDf_y):
    clf = models.load_model(f)
    clf.summary()

    h = clf.fit(trainDf_x, trainDf_y, validation_split=0.25, **p)
    h = pd.DataFrame(h.history)
    h['epoch'] = h.index + 1
    h = h.round(3)
    return (h, clf)

def _evaluateClassifier(clf, testDf):
    (testDf_x, testDf_y) = getInputOutputVariables(testDf)
    testResults = _getPredictions(clf, testDf)

    ## Testing scores
    testScores = clf.evaluate(testDf_x, testDf_y)
    testScores = [[t] for t in testScores]
    testScores = pd.DataFrame(dict(zip(clf.metrics_names, testScores)))
    testScores = testScores.round(3)
    return (testResults, testScores)

def _getPredictions(clf, testDf):
    ## Prepare data
    testId = testDf.iloc[:, :4].copy()
    testCount = testDf.iloc[:, 4:]
    testCount.columns = testCount.columns.astype(str)

    ## Make predictions
    prob = clf.predict(testCount)
    pred = np.where(prob > 0.5, 1, 0)
    clsStr = np.where(pred == 1, 'Synthetic', 'Not synthetic')

    ## Keep track of the results
    testId['model']              = 'KerasClassifier'
    testId['predicted_classIdx'] = pred
    testId['predicted_class']    = clsStr
    testId['predicted_prob']     = prob
    return testId

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
