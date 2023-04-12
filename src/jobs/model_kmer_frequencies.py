#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
from pathlib import Path

# External imports
import numpy as np
import pandas as pd

# Internal imports
from ..shared.kmer.common import *
from ..shared.common import *
from ..shared import io

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def main(iDirs, modelFiles, oFile):
    from tensorflow.keras import models

    ## Prepare data
    kmerDf = io.kmer.read(*iDirs)
    kmerDf = _prepareData(kmerDf)

    ## Read models and make predictions
    clfs = (models.load_model(f) for f in modelFiles)
    results = [_getPredictions(clf, kmerDf) for clf in clfs]
    results = pd.concat(results).reset_index(drop=True)

    ## Write the tables to disk
    p = Path(oFile)
    p.mkdir(parents=True, exist_ok=True)
    results.to_csv(oFile, sep='\t', index=False)

#------------------- Private Classes & Functions ------------#

def _getPredictions(clf, kmerDf):
    ## Prepare data
    kmerId = kmerDf[['seqId', 'seqLen']].copy()
    kmerCount = kmerDf.iloc[:, 2:]
    kmerCount.columns = kmerCount.columns.astype(str)

    ## Make predictions
    prob = clf.predict(kmerCount)
    pred = np.where(prob > 0.5, 1, 0)
    clsStr = np.where(pred == 1, 'Synthetic', 'Not synthetic')

    ## Keep track of the results
    kmerId['model']              = 'KerasClassifier'
    kmerId['predicted_classIdx'] = pred
    kmerId['predicted_class']    = clsStr
    kmerId['predicted_prob']     = prob
    return kmerId

def _prepareData(kmerDf):
    def _groupRevComp(df):
        f = lambda x: '-'.join(sorted([x, seqToRevComp(x)]))
        df['kmer'] = df['kmer'].apply(f, meta=df['kmer'])
        df = df.groupby(['seqId', 'seqLen', 'kmer']).sum() * 2
        df = df.reset_index()
        return df

    def _preprocessCounts(kmerCount):
        from sklearn import preprocessing
        kmerCount = kmerCount.fillna(0)
        kmerCount = kmerCount.divide(kmerCount.sum(axis=1), axis=0)
        kmerCount[kmerCount.columns] = preprocessing.normalize(kmerCount)
        return kmerCount

    ## Rearrange the table (i.e., vectorise counts)
    kmerDf = _groupRevComp(kmerDf)
    kmerDf['kmer'] = kmerDf['kmer'].str.split('-', n=1, expand=True)[0]

    ## Try to load everything into memory
    kmerId = kmerDf[['seqId', 'seqLen']].drop_duplicates()
    kmerId = kmerId.set_index('seqId')
    kmerId = kmerId.compute()

    kmerCount = kmerDf.categorize(columns=['kmer'])
    kmerCount = kmerCount.pivot_table(index='seqId', columns='kmer', values='count')
    kmerCount.columns = [seqToInt(c) for c in kmerCount.columns]
    kmerCount = kmerCount.compute()

    ## Apply any transformations that are not Train-Test specific.
    ## Transformations with native python are more efficient than Spark
    ## assuming we can load the data into memory.
    kmerCount = _preprocessCounts(kmerCount)

    ## Create a single dataframe
    kmerDf = pd.concat([kmerId, kmerCount], axis=1)
    kmerDf = kmerDf.reset_index()
    return kmerDf

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
