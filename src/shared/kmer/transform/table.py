#!/bin/python

#------------------- Description & Notes --------------------#

'''
We can reduce the number of K-mers by:
* Removing correlated columns
* Removing constant columns

We can reduce the number of samples by:
* Removing duplicate rows (For relatively short K-mers, i.e., K < 5)
  This is actually good for highly related sequences. However, for distantly
  related sequences, this won't actually do much because the counts become
  more unique as the kmer length increases.
'''

#------------------- Dependencies ---------------------------#

# Standard library imports
import zlib

# External imports
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from scipy import sparse

# Internal imports
from .. import analyse
from ..common import *

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def removeConstantCounts(kmerDf):
    ## Find Kmer containing the same value across all rows
    ## (i.e., their standard deviation == 0)
    w = Window.partitionBy(KMER_COL_NAME)
    kmerDf = kmerDf.withColumn('std', F.stddev(COUNT_COL_NAME).over(w))
    kmerDf = kmerDf.filter(F.col('std') != 0)
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    return kmerDf

def removeCorrelatedCounts(kmerDf, corr=0.90):
    ## Find Kmers that are highly correlated. We don't really care
    ## which Kmer is removed, but we'll remove the first one we encounter
    kmerCorrDf = analyse.feature.getCorrelationTable(kmerDf)
    kmerCorrDf = kmerCorrDf.filter(F.col('corr') > corr)
    toRemove = (
        kmerCorrDf
        .select('kmer_1').distinct()
        .withColumnRenamed('kmer_1', 'kmerIdx')
    )

    ## Anti-join to get the rows we want
    kmerDf = insertKmerIdxCol(kmerDf)
    kmerDf = kmerDf.join(toRemove, on='kmerIdx', how="left_anti")
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    return kmerDf

def removeSparseCounts(kmerDf, threshold=0.8):
    ## Find Kmers that absent in more than 80% of the samples.
    nSeqs = kmerDf.select('seqId').distinct().count()
    w = Window.partitionBy(KMER_COL_NAME)
    kmerDf = kmerDf.withColumn('prop', F.count(KMER_COL_NAME).over(w) / F.lit(nSeqs))
    kmerDf = kmerDf.filter(F.col('prop') > threshold)
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    return kmerDf

def removeOneCounts(kmerDf):
    ## Find Kmers that only occur once.
    ## These may be caused by sequencing errors.
    kmerDf = kmerDf.filter(F.col(COUNT_COL_NAME) != 1)
    return kmerDf

def removeRepetitiveKmers(kmerDf, threshold=0):
    ## Find repetitive Kmers; these are considered 'low complexity'.
    ## Complexity is based on the size (in bytes) of the Kmer before
    ## and after compression. See Sims et al. (2009).
    f = lambda x: (len(zlib.compress(x.encode())) - len(x.encode()))
    f = F.udf(f, T.IntegerType())
    kmerDf = kmerDf.withColumn('complexity', f(KMER_COL_NAME))
    kmerDf = kmerDf.filter(kmerDf.complexity > threshold)
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    return kmerDf

def removeShortSequences(kmerDf, n=1):
    ## Find relatively short sequences (i.e., len(Seq) < threshold)
    kLen    = len(kmerDf.select(KMER_COL_NAME).first()[0])
    minLen  = (4 ** kLen) * n
    kmerDf = kmerDf.filter(kmerDf.seqLen > minLen)
    return kmerDf

def insertZeroCounts(kmerDf):
    ## Collect Kmers in each sequence
    x = (
        kmerDf
        .groupby(KMERDF_SEQINFO_COL_NAMES)
        .agg(F.collect_list(KMER_COL_NAME).alias(KMER_COL_NAME))
    )

    ## Collect Kmers across all sequences
    y = (
        kmerDf
        .select(
            F.explode(F.collect_set(SEQID_COL_NAME)).alias(SEQID_COL_NAME),
            F.collect_set(KMER_COL_NAME).alias('oKmers')
        )
    )

    ## Join the tables (try avoid broadcasting) and find Kmers
    ## in some but not all sequences
    z = x.join(y, on=SEQID_COL_NAME, how='left')
    z = (
        z
        .select(
            *KMERDF_SEQINFO_COL_NAMES,
            F.explode(F.array_except('oKmers', KMER_COL_NAME)).alias(KMER_COL_NAME),
            F.lit(0).alias(COUNT_COL_NAME)
        )
    )

    ## Add them to the original table
    kmerDf = kmerDf.union(z)
    return kmerDf

def insertKmerIdxCol(kmerDf):
    f = F.pandas_udf(lambda x: x.apply(seqToInt), T.IntegerType())
    kmerDf = kmerDf.withColumn('kmerIdx', f(KMER_COL_NAME))
    return kmerDf

def toPdfRdd(kmerSdf):
    ## Convert the Spark DataFrame into a Spark RDD of Pandas DataFrames
    ## We assume that the columns are standard types and not Spark Vectors
    cols = kmerSdf.schema.names
    f    = lambda x: [pd.DataFrame(list(x), columns=cols)]
    g    = lambda x: len(x) != 0
    kmerPdfRdd = kmerSdf.rdd.mapPartitions(f).filter(g)
    return kmerPdfRdd

def splitPdf(kmerPdf):
    seqInfoColNames = set(kmerPdf).difference(set(KMERDF_KMERCOUNT_COL_NAMES))

    f = lambda x: dict(zip(*x))
    countCols = kmerPdf[KMERDF_KMERCOUNT_COL_NAMES].apply(f, axis=1)
    countCols = countCols.apply(pd.Series)
    infoCols  = kmerPdf[seqInfoColNames]
    return (infoCols, countCols)

def toDriver(kmerDf):
    def _printMemoryUsage(m):
        BYTES_TO_MB_DIV = 0.000001
        mem = (m.data.nbytes + m.indptr.nbytes + m.indices.nbytes) * BYTES_TO_MB_DIV
        print("Memory usage is " + str(mem) + " MB")

    ## Split the data into (K, V) pair
    kmerDf.persist()
    kmerId    = kmerDf.drop(KMER_COL_NAME).toPandas()
    kmerCount = kmerDf.select(KMER_COL_NAME)

    ## Collect SparseVectors into a scipy matrix
    ## Seem fairly memory efficient (for 9mers), although it would
    ## be entirely dependent on the number of Kmers
    kmerCount = (
        kmerCount.rdd
        .map(lambda x: sparse.csr_matrix(x[0].toArray()))
        .mapPartitions(lambda x: (yield sparse.vstack(x)))
        .reduce(lambda x, y: sparse.vstack([x, y]))
    )
    _printMemoryUsage(kmerCount)
    return (kmerId, kmerCount)

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
