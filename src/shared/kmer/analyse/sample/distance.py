#!/bin/python

#------------------- Description & Notes --------------------#

'''
D2S and D2Star seem to be the best performing metric for measuring
dissimilarity between Kmer frequencies.

In terms of definition they are pretty similar and D2Star reportedly
performs better than D2S. However, D2S appears to be more widely
used because its easier to compute.
'''

'''
Spark allows us to do very large and scalable pairwise comparisons
However, doing anything useful with a super large square distance matrix
will be very problematic.

One thing with Spark is that it can't seem to handle wide data
very well (at least in the case of the DataFrames). This can get a
bit problematic when we want to do pairwise comparisons as the
square matrices become MASSIVE.

Alternatively, we could find out the distances for only the X closest points
since we're predominantly interested in points that are relatively close.
Therefore, we'll end up with a table of idxs instead of distances.
'''

'''
We can calculate pairwise distances using the idea from the following link.
It seems to work well if we don't need the whole matrix (but not what we want)
https://medium.com/@rantav/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e
'''

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from sklearn.metrics import pairwise_distances

# Internal imports
from .... import kmer
from ... import transform
from .common import *

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def getTraditionalDistances(kmerDfX, kmerDfY=None, metric='euclidean'):
    def _prepareData(kmerDfX, kmerDfY):
        nParts  = kmerDfX.rdd.getNumPartitions()
        kmerDfX = transform.counts.toVector(kmerDfX)
        kmerDfX = kmerDfX.select(kmer.SEQID_COL_NAME, kmer.KMER_COL_NAME)
        kmerDfX = kmerDfX.coalesce(nParts)

        if (kmerDfY is not None):
            kmerDfY = transform.counts.toVector(kmerDfY)
            kmerDfY = kmerDfY.select(kmer.SEQID_COL_NAME, kmer.KMER_COL_NAME)
            kmerDfY = kmerDfY.coalesce(nParts)

        return (kmerDfX, kmerDfY)

    ## Prepare the data
    (kmerDfX, kmerDfY) = _prepareData(kmerDfX, kmerDfY)

    ## Find all pairs depending on the DataFrames available
    kmerDistDf = getPairs(kmerDfX, kmerDfY)

    ## Calculate the distance for each pair
    f = lambda x, y: float(pairwise_distances([x], [y], metric=metric))
    f = F.udf(f, T.FloatType())
    kmerDistDf = kmerDistDf.select(
        F.col('l.SeqId').alias('seqId_1'),
        F.col('r.SeqId').alias('seqId_2'),
        f('l.kmer', 'r.kmer').alias('distance')
    )
    return kmerDistDf

def getAdjustedDistances(obsKmerDf, expKmerDf, metric):
    def _prepareData(obsKmerDf, expKmerDf):
        nParts = obsKmerDf.rdd.getNumPartitions()
        obsKmerDf = transform.counts.toVector(obsKmerDf)
        obsKmerDf = (
            obsKmerDf
            .select(
                kmer.SEQID_COL_NAME,
                F.col(kmer.KMER_COL_NAME).alias('kmer_obs')
            )
        )
        obsKmerDf = obsKmerDf.coalesce(nParts)

        nParts = expKmerDf.rdd.getNumPartitions()
        expKmerDf = transform.counts.toVector(expKmerDf)
        expKmerDf = (
            expKmerDf
            .select(
                kmer.SEQID_COL_NAME,
                F.col(kmer.KMER_COL_NAME).alias('kmer_exp')
            )
        )
        expKmerDf = expKmerDf.coalesce(nParts)

        ## Join the tables
        kmerDf = obsKmerDf.join(expKmerDf, on=[kmer.SEQID_COL_NAME], how='left')
        return kmerDf

    ## Create a single table containing the Obs and Exp counts
    ## This may takes some time due to the vectorisation
    kmerDistDf = _prepareData(obsKmerDf, expKmerDf)

    ## Crossjoin the table to find all the pairs we want and
    ## ensure that we only need to process the upper triangle
    kmerDistDf = getUpperTriangle(kmerDistDf)

    ## Calculate the distance for each pair
    f = lambda x, y: float(metric(x, y))
    f = F.udf(f, T.FloatType())
    kmerDistDf = kmerDistDf.select(
        F.col('l.SeqId').alias('seqId_1'),
        F.col('r.SeqId').alias('seqId_2'),
        f(
            F.array('l.kmer_obs', 'l.kmer_exp'),
            F.array('r.kmer_obs', 'r.kmer_exp')
        ).alias('distance')
    )
    return kmerDistDf

def scale(kmerDistDf, scale=(0, 100)):
    va   = VectorAssembler(inputCols=['distance'], outputCol="v")
    mms  = MinMaxScaler(min=scale[0], max=scale[1], inputCol="v", outputCol="s")
    pipe = Pipeline(stages=[va, mms])

    ## Perform scaling pipeline
    m          = pipe.fit(kmerDistDf)
    kmerDistDf = m.transform(kmerDistDf)

    ## Reorganise and cleanup
    f = lambda x: float(x[0])
    f = F.udf(f, T.FloatType())
    kmerDistDf = kmerDistDf.select(
        'seqId_1', 'seqId_2', f('s').alias('distance')
    )
    return kmerDistDf

#------------------- Private Classes & Functions ------------#

def _D2(x, y):
    return np.sum(x * y)

def _d2(x, y):
    D2 = _D2(x, y)
    sq_1_sum = np.sum(x ** 2)
    sq_2_sum = np.sum(y ** 2)

    denom = np.sqrt(sq_1_sum) * np.sqrt(sq_2_sum)
    d2    = 0.5 * (1.0 - (sumNum / denom))
    return d2

def _d2S(x, y):
    xObs = x[0].toArray()
    xExp = x[1].toArray()
    yObs = y[0].toArray()
    yExp = y[1].toArray()

    diff1 = xObs - xExp
    diff2 = yObs - yExp
    sq1   = diff1 * diff1
    sq2   = diff2 * diff2
    denom = np.sqrt((sq1 + sq2))

    sumNum   = ((diff1 * diff2) / denom).sum()
    sumDenom = (np.sqrt((sq1 / denom)) * np.sqrt((sq2 / denom))).sum()
    d2S      = 0.5 * (1.0 - (sumNum / sumDenom))
    return d2S

def _d2Star(x, y):
    xObs = x[0].toArray()
    xExp = x[1].toArray()
    yObs = y[0].toArray()
    yExp = y[1].toArray()

    diff1  = xObs - xExp
    diff2  = yObs - yExp
    num    = (diff1 * diff2) / np.sqrt((xExp * yExp))
    sqDiv1 = (diff1 * diff1) / xExp
    sqDiv2 = (diff2 * diff2) / yExp

    sumNum   = num.sum()
    sumDenom = np.sqrt(sqDiv1.sum()) * np.sqrt(sqDiv2.sum())
    d2Star   = 0.5 * (1.0 - (sumNum / sumDenom))
    return d2Star

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
