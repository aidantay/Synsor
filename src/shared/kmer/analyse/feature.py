#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.functions import vector_to_array
from pyspark.ml.stat import Summarizer
from pyspark.ml.stat import Correlation
from scipy.stats.stats import pearsonr

# Internal imports
from ... import kmer
from .. import transform
from .sample import getUpperTriangle

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def getStatisticTable(kmerDf, asVector=True):
    if (asVector):
        ## Convert the counts into vector (only if they need to)
        if (dict(kmerDf.dtypes)[kmer.KMER_COL_NAME] == 'string'):
            nParts = kmerDf.rdd.getNumPartitions()
            kmerDf = transform.counts.toVector(kmerDf)
            kmerDf = kmerDf.coalesce(nParts)

        ## For each K-mer, calculate the mean and standard deviation.
        ## I suspect this is only possible when vectors contain less than
        ## 65,536 (i.e., 2^16) columns.
        kmerStatDf = (
            kmerDf
            .select(
                vector_to_array(Summarizer.mean(F.col(kmer.KMER_COL_NAME))).alias('mean'),
                vector_to_array(Summarizer.std(F.col(kmer.KMER_COL_NAME))).alias('std')
            )
        )

        ## Reorganise the table table
        f = lambda x: [i for i in range(len(x))]
        f = F.udf(f, T.ArrayType(T.IntegerType()))
        kmerStatDf = (
            kmerStatDf
            .withColumn('idx', f('mean'))
            .withColumn('tmp', F.explode(F.arrays_zip('idx', 'mean', 'std')))
            .select(
                F.col('tmp.idx').alias(kmer.KMER_COL_NAME),
                F.col('tmp.mean').alias('mean'),
                F.col('tmp.std').alias('std')
            )
        )

    else:
        ## Prepare data
        kmerDf = transform.table.insertZeroCounts(kmerDf)

        ## For each K-mer, calculate the mean and standard deviation.
        ## Theoretically, this is slower but more scalable than the above
        kmerStatDf = (
            kmerDf
            .groupby(kmer.KMER_COL_NAME)
            .agg(
                F.mean(kmer.COUNT_COL_NAME).alias("mean"),
                F.stddev(kmer.COUNT_COL_NAME).alias("std")
            )
        )

    return kmerStatDf

def getCorrelationTable(kmerDf, asVector=True):
    if (asVector):
        ## Convert the counts into vector (only if they need to)
        if (dict(kmerDf.dtypes)[kmer.KMER_COL_NAME] == 'string'):
            nParts = kmerDf.rdd.getNumPartitions()
            kmerDf = transform.counts.toVector(kmerDf)
            kmerDf = kmerDf.coalesce(nParts)

        ## Perform correlation
        ## This is only possible when the vectors contain less than
        ## 65,536 (i.e., 2^16) columns.
        corrM = Correlation.corr(kmerDf, kmer.KMER_COL_NAME, method='pearson')

        ## Unpack the matrix
        corrRdd = corrM.rdd.flatMap(lambda x: x[0].toArray().tolist())
        corrRdd = corrRdd.repartition(kmerDf.rdd.getNumPartitions())
        corrRdd = (
            corrRdd
            .zipWithIndex()
            .map(lambda x: (x[1], x[0]))
            .flatMapValues(enumerate)
        )
        corrRdd = corrRdd.repartition(kmerDf.rdd.getNumPartitions() * 2)

        ## Ensure we only get the upper triangle
        corrRdd = (
            corrRdd
            .map(lambda x: (x[0], *x[1]))
            .filter(lambda x: (x[0] <= x[1]))
        )

        kmerCorrDf = corrRdd.toDF(['kmer_1', 'kmer_2', 'corr'])

    else:
        ## Prepare data
        nParts = kmerDf.rdd.getNumPartitions()
        kmerDf = transform.table.insertZeroCounts(kmerDf)
        kmerDf = (
            kmerDf
            .groupby(kmer.KMER_COL_NAME)
            .agg(F.collect_list(kmer.COUNT_COL_NAME).alias(kmer.COUNT_COL_NAME))
        )
        kmerDf = kmerDf.coalesce(nParts)
        kmerDf.cache()

        ## Insert a column containing the row number so that we only need
        ## to process the upper triangle and crossjoin the table to find all
        ## the pairs we want
        kmerCorrDf = getUpperTriangle(kmerDf)
        kmerCorrDf = kmerCorrDf.repartitionByRange(nParts ** 2, F.col('l.kmer'))

        ## Calculate the correlation for each pair
        f = lambda x, y: float(pearsonr(x, y)[0])
        f = F.udf(f, T.FloatType())
        kmerCorrDf = kmerCorrDf.select(
            F.col('l.kmer').alias('kmer_1'),
            F.col('r.kmer').alias('kmer_2'),
            f('l.count', 'r.count').alias('corr')
        )

    return kmerCorrDf

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
