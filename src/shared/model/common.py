#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import numpy as np
import pandas as pd
import dask.bag as db
from scipy import sparse
from sklearn import preprocessing
from sklearn import model_selection

# Internal imports
from ..kmer.common import *
from ..common import *

#------------------- Constants ------------------------------#


#------------------- Public Classes & Functions -------------#

def readClassInfo(clsFiles):
    clsDf = [pd.read_csv(f, sep='\t') for f in clsFiles]
    clsDf = pd.concat(clsDf)
    clsDf['seqId'] = clsDf['seqAcc'] + ' ' + clsDf['seqDesc']
    clsDf = clsDf[['seqId', 'class']].copy()
    clsDf = clsDf.set_index('seqId')
    return clsDf

def prepareData(kmerDf, clsDf=None):
    def _getExpectedFeatures(df):
        kLen = len(df.head(1)['kmer'][0])
        cols = db.from_sequence(getExpectedSequences(kLen))
        nParts = cols.npartitions

        cols = (
            cols
            .map(lambda x: [seqToRevComp(x), x]).map(sorted).pluck(0)
            .distinct()
        )
        cols = (
            cols
            .repartition(nParts)
            .map(seqToInt)
        )
        cols = sorted(cols.compute())
        return cols

    def _groupRevComp(df):
        nParts = df.npartitions
        f = lambda x: '-'.join(sorted([x, seqToRevComp(x)]))
        df['kmer'] = df['kmer'].apply(f, meta=df['kmer'])
        df = df.groupby(['seqId', 'seqLen', 'kmer'], sort=False).sum(split_out=nParts) * 2
        df = df.reset_index()
        return df

    def _toList(df):
        nParts = df.npartitions
        df = df.groupby(['seqId', 'seqLen']).agg({'kmer':'list', 'count':'list'}, split_out=nParts)
        return df

    def _getSparseCounts(df, vSize):
        def _toSparseArray(x, vSize):
            col  = x[2]
            row  = np.zeros(len(col))
            data = x[3]
            arr = sparse.csr_array((data, (row, col)), shape=(1, vSize))
            return arr

        df = df.reset_index().to_bag()
        df = df.map(lambda x: _toSparseArray(x, vSize))
        df = df.reduction(sparse.vstack, sparse.vstack)
        return df

    def _preprocessCounts(kmerCount):
        kmerCount = (kmerCount / kmerCount.sum(axis=1))
        kmerCount = preprocessing.normalize(np.asarray(kmerCount))
        kmerCount = pd.DataFrame(kmerCount)
        return kmerCount

    def _insertClassIdxCol(df, col='class'):
        ## The model we are creating will be classifying sequences as
        ## synthetic or not synthetic. In which case, we assign
        ## synthetic sequences (i.e., vector sequences) as the positive
        ## class (i.e., synthetic = 1) and all other sequences as the
        ## negative class (i.e., not synthetic = 0).
        idxCol = col + 'Idx'
        cond = (df[col].str.lower() == 'vector')
        df.loc[cond, col]     = 'Synthetic'
        df.loc[cond, idxCol]  = 1
        df.loc[~cond, col]    = 'Not synthetic'
        df.loc[~cond, idxCol] = 0
        return df

    ## Keep track of the Kmers (i.e., features)
    cols = _getExpectedFeatures(kmerDf)
    vSize = cols[-1] + 1

    ## Rearrange the table (i.e., vectorise counts)
    kmerDf = _groupRevComp(kmerDf)
    print(kmerDf)
    kmerDf['kmer'] = kmerDf['kmer'].str.split('-', n=1, expand=True)[0]
    kmerDf['kmer'] = kmerDf['kmer'].apply(seqToInt, meta=kmerDf['kmer'])
    kmerDf = _toList(kmerDf)
    print(kmerDf)

    ## Try to load everything into memory
    kmerId = kmerDf.reset_index()[['seqId', 'seqLen']].compute()
    kmerCount = _getSparseCounts(kmerDf, vSize).compute()
    print(kmerId)
    print(kmerCount)

    ## Apply any transformations that are not Train-Test specific.
    ## Transformations with native python are more efficient than Spark
    ## assuming we can load the data into memory.
    kmerCount = _preprocessCounts(kmerCount)

    ## Ensure that the columns are consistent across training and testing data.
    ## This will mean that some columns will only contain 0s, which hopefully
    ## won't be an issue for the neural network / preprocessing steps.
    kmerCount = kmerCount.reindex(columns=cols).fillna(0)

    ## If applicable, read and add class info
    if (clsDf is not None):
        kmerId = kmerId.merge(clsDf, on='seqId', how='left')
        kmerId = _insertClassIdxCol(kmerId)

    ## Create a single dataframe
    ## Dask doesn't reset indexes entirely...Not sure why...
    kmerId = kmerId.reset_index(drop=True)
    kmerDf = pd.concat([kmerId, kmerCount], axis=1)
    return kmerDf

def shuffleAndSplitData(kmerDf):
    (trainDf, testDf) = model_selection.train_test_split(kmerDf, test_size=0.2, random_state=42)
    trainDf = trainDf.reset_index(drop=True)
    testDf  = testDf.reset_index(drop=True)
    print(kmerDf[['class', 'classIdx']].drop_duplicates())
    print(trainDf.shape)
    print(trainDf['classIdx'].value_counts().sort_values())
    print(testDf.shape)
    print(testDf['classIdx'].value_counts().sort_values())
    return (trainDf, testDf)

def getInputOutputVariables(train_or_test):
    x = train_or_test.iloc[:, 4:]
    x.columns = x.columns.astype(str)
    y = train_or_test['classIdx']
    return (x, y)

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
