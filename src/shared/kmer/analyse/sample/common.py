#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from scipy.spatial.distance import squareform

# Internal imports

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def getPairs(kmerSdfX, kmerSdfY=None):
    if (kmerSdfY is None):
        kmerSdf = getUpperTriangle(kmerSdfX)

    else:
        kmerSdf = kmerSdfX.alias('l').crossJoin(kmerSdfY.alias('r'))

    return kmerSdf

def getUpperTriangle(kmerSdf):
    ## Insert a column containing the row number so that we only need
    ## to process the upper triangle and crossjoin the table to find all
    ## the pairs we want
    kmerSdf = kmerSdf.withColumn('rId', F.monotonically_increasing_id())
    cond = F.col('l.rId') <= F.col('r.rId')
    kmerSdf = kmerSdf.alias('l').join(kmerSdf.alias('r'), on=cond, how='outer')
    return kmerSdf

def tableToSymMatrix(kmerDistDf):
    ## Construct a list of distances from a sorted dictionary
    rows  = kmerDistDf.to_dict('records')
    rows  = {tuple(sorted([r['seqId_1'], r['seqId_2']])):r['distance'] for r in rows}
    dists = [dist[1] for dist in sorted(rows.items())]

    ## Get all the IDs and construct the matrix
    ids = list(pd.concat([kmerDistDf['seqId_1'], kmerDistDf['seqId_2']]).unique())
    kmerDistDf = pd.DataFrame(squareform(dists), index=sorted(ids), columns=sorted(ids))
    return kmerDistDf

def matrixToSymMatrix(kmerDistDf, useMax=True):
    triu = np.triu(kmerDistDf)
    tril = np.tril(kmerDistDf)

    ## Compare the triangles and get the maximum value for each pair
    newTril = tril.T
    newTriu = np.maximum(triu, newTril) if useMax else np.minimum(triu, newTril)

    ## Reconstruct the matrix
    m = newTriu + newTriu.T
    np.fill_diagonal(m, 0)
    kmerDistDf = pd.DataFrame(m, columns=kmerDistDf.columns,
        index=kmerDistDf.index)

    return kmerDistDf

def getZScore(kmerStatsDf, colName):
    ## Calculate the mean and standard deviation
    df = kmerStatsDf.select(F.mean(colName),
        F.stddev(colName)).collect()[0]
    (mean, std) = df

    ## Z-Score calculation
    f = lambda x: (x - mean) / std
    f = F.pandas_udf(f, T.FloatType())
    kmerStatsDf = kmerStatsDf.withColumn('ZScore', f(F.col(colName)))
    return kmerStatsDf

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
