#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.linalg import Vectors

# Internal imports
from .kmerseqs import appendRevComp
from .kmerseqs import toInt
from .table import toDriver
from ..schema import *
from .. import util

#------------------- Constants ------------------------------#

#------------------- Classes & Functions --------------------#

def toProbabilities(kmerDf):
    kmerDf = kmerDf.withColumn(COUNT_COL_NAME,
        F.col(COUNT_COL_NAME).cast(T.FloatType()))
    kmerDf = (kmerDf.groupby(KMERDF_SEQINFO_COL_NAMES)
        .applyInPandas(_toProbabilities, schema=kmerDf.schema))
    return kmerDf

def toNormalised(kmerDf):
    ## Normalise counts according to Wang et al. 2005
    ## This is done by multiplying the probabilities with
    ## the total number of possible Kmers
    kmerSeq = kmerDf.select(KMER_COL_NAME).first()[0]
    t = util.kmer.getExpectedTotal(kmerSeq)

    kmerDf = toProbabilities(kmerDf)
    kmerDf = kmerDf.withColumn(COUNT_COL_NAME, F.col(COUNT_COL_NAME) * t)
    return kmerDf

def toL2Normalised(kmerDf):
    kmerDf = kmerDf.withColumn(COUNT_COL_NAME,
        F.col(COUNT_COL_NAME).cast(T.FloatType()))
    kmerDf = (kmerDf.groupby(KMERDF_SEQINFO_COL_NAMES)
        .applyInPandas(_toL2Normalised, schema=kmerDf.schema))
    return kmerDf

def explode(kmerDf):
    gCols = kmerDf.schema.names
    gCols.remove(KMER_COL_NAME)
    gCols.remove(COUNT_COL_NAME)

    ## (ID, [Kmer], [Count]) => (ID, Kmer, Count)
    f = F.explode(F.arrays_zip(KMER_COL_NAME, COUNT_COL_NAME))
    kmerDf = (
        kmerDf.withColumn('tmp', f)
        .select(
            *gCols,
            F.col('tmp.kmer').alias(KMER_COL_NAME),
            F.col('tmp.count').alias(COUNT_COL_NAME)
        )
    )
    return kmerDf

def toDict(kmerDf):
    ## (ID, Kmer, Count) => (ID, [Kmer], [Count])
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    kmerDf = toList(kmerDf, isSorted=False)

    ## (ID, Kmer, Count) => (ID, {Kmer => Count})
    f = F.map_from_arrays(KMER_COL_NAME, COUNT_COL_NAME)
    kmerDf = kmerDf.select(*KMERDF_SEQINFO_COL_NAMES, f.alias(KMER_COL_NAME))
    return kmerDf

def toList(kmerDf, isSorted=True):
    def _sortCounts(kmerPdf):
        ## Convert the counts into a sorted list based on the Kmers
        kmerPdf = kmerPdf.sort_values(by=[KMER_COL_NAME])
        counts  = kmerPdf[COUNT_COL_NAME].tolist()
        kmer    = kmerPdf[KMER_COL_NAME].tolist()

        newKmerPdf = kmerPdf.iloc[:1][KMERDF_SEQINFO_COL_NAMES]
        newKmerPdf[COUNT_COL_NAME] = [counts] * 1
        newKmerPdf[KMER_COL_NAME]  = [kmer] * 1
        return newKmerPdf

    ## (ID, Kmer, Count) => (ID, [Kmer], [Count])
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    if (isSorted):
        schema  = getSchema(countsAsInt=False)
        kmerDf = (kmerDf.groupby(KMERDF_SEQINFO_COL_NAMES)
            .applyInPandas(_sortCounts, schema=getSchema()))

    else:
        kmerDf = (
            kmerDf.groupby(KMERDF_SEQINFO_COL_NAMES)
            .agg(
                F.collect_list(KMER_COL_NAME).alias(KMER_COL_NAME),
                F.collect_list(COUNT_COL_NAME).alias(COUNT_COL_NAME)
            )
        )

    return kmerDf

def toVector(kmerDf, toDense=False):
    ## Kmer => Idx
    kmerDf  = kmerDf.select(KMERDF_COL_NAMES)
    kmerSeq = kmerDf.select(KMER_COL_NAME).first()[0]
    kmerDf  = kmerDf.withColumn(KMER_COL_NAME, F.split(KMER_COL_NAME, '-').getItem(0))
    kmerDf  = toInt(kmerDf)
    if ('-' in kmerSeq):
        ## Re-index KmerIdx so that they range from (0, 4^N/2)
        w = Window.orderBy(KMER_COL_NAME)
        kmerDf = kmerDf.withColumn(KMER_COL_NAME, F.dense_rank().over(w) - 1)

    ## Get the size of the vector
    vSize = kmerDf.select(F.max(KMER_COL_NAME)).collect()[0][0] + 1

    ## (ID, KmerIdx, Count) => (ID, {KmerIdx => Count})
    kmerDf = toDict(kmerDf)

    ## Convert the counts into SparseVectors
    f = F.udf(lambda x: Vectors.sparse(vSize, x), VectorUDT())
    kmerDf = kmerDf.withColumn(KMER_COL_NAME, f(KMER_COL_NAME))

    ## If applicable, convert the counts into DenseVectors
    if (toDense):
        f = F.udf(lambda x: Vectors.dense(x.toArray()), VectorUDT())
        kmerDf = kmerDf.withColumn(KMER_COL_NAME, f(KMER_COL_NAME))

    ## Vectorising loses the Kmer string information. We probably don't need
    ## them, but if we do, then we can convert the vector indices into Kmers.
    return kmerDf

def groupBy(kmerDf, *cols, func=F.sum, sep=' '):
    ## Sum the counts for each Kmer
    x = (kmerDf.groupby(*cols, KMER_COL_NAME)
        .agg(func(COUNT_COL_NAME).alias(COUNT_COL_NAME)))

    ## Sum the lengths of each sequence
    y = (
        kmerDf.select(*cols, *KMERDF_SEQINFO_COL_NAMES).distinct()
        .groupby(*cols)
        .agg(func(SEQLEN_COL_NAME).alias(SEQLEN_COL_NAME))
    )

    ## Join the tables and format the columns
    if (len(cols) == 0):
        kmerDf = x.crossJoin(y)
        kmerDf = kmerDf.withColumn(SEQID_COL_NAME, F.lit('GroupedCounts'))
        kmerDf = kmerDf.select(*KMERDF_COL_NAMES)

    else:
        kmerDf = x.join(y, on=[*cols], how='left')
        kmerDf = kmerDf.withColumn(SEQID_COL_NAME, F.concat_ws(sep, *cols))
        kmerDf = kmerDf.select(*KMERDF_COL_NAMES)

    return kmerDf

def groupRevComp(kmerDf):
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    kmerDf = appendRevComp(kmerDf)
    kmerDf = (
        kmerDf.groupby(*KMERDF_SEQINFO_COL_NAMES, KMER_COL_NAME)
        .agg(F.sum(COUNT_COL_NAME).alias(COUNT_COL_NAME))
        .withColumn(COUNT_COL_NAME, F.col(COUNT_COL_NAME) * 2)
    )
    return kmerDf

def _toProbabilities(kmerPdf):
    total = kmerPdf[COUNT_COL_NAME].sum()
    kmerPdf[COUNT_COL_NAME] = kmerPdf[COUNT_COL_NAME] / total
    kmerPdf[COUNT_COL_NAME] = kmerPdf[COUNT_COL_NAME]
    return kmerPdf

def _toL2Normalised(kmerPdf):
    from sklearn import preprocessing
    counts = kmerPdf[COUNT_COL_NAME].to_numpy().reshape(1, -1)
    kmerPdf[COUNT_COL_NAME] = preprocessing.normalize(counts)[0]
    return kmerPdf

#------------------- Main -----------------------------------#

