#!/bin/python

#------------------- Description & Notes --------------------#

'''
We can reduce the number of K-mers by:
* Clustering reverse complement counts (For relatively short K-mers, i.e. K < 5)

We can reduce the number of samples by:
* Clustering counts of related sequences
'''

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.linalg import Vectors
from sklearn import preprocessing

# Internal imports
from ..common import *
from .table import insertKmerIdxCol

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

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
    kmer = kmerDf.select(KMER_COL_NAME).first()[0]
    t    = getExpectedTotal(kmer)

    kmerDf = toProbabilities(kmerDf)
    kmerDf = kmerDf.withColumn(COUNT_COL_NAME, F.col(COUNT_COL_NAME) * t)
    return kmerDf

def toL2Normalised(kmerDf, withPandas=True):
    if (withPandas):
        kmerDf = kmerDf.withColumn(COUNT_COL_NAME,
            F.col(COUNT_COL_NAME).cast(T.FloatType()))
        kmerDf = (kmerDf.groupby(KMERDF_SEQINFO_COL_NAMES)
            .applyInPandas(_toL2Normalised, schema=kmerDf.schema))

    else:
        normalizer = Normalizer(inputCol=KMER_COL_NAME, outputCol='L2Norm')
        kmerDf = normalizer.transform(kmerDf)
        kmerDf = kmerDf.drop(KMER_COL_NAME)
        kmerDf = kmerDf.withColumnRenamed('L2Norm', KMER_COL_NAME)

    return kmerDf

def explode(kmerDf):
    ## (ID, [Kmer], [Count], *) => (ID, Kmer, Count, *)
    gCols = kmerDf.schema.names
    gCols.remove(KMER_COL_NAME)
    gCols.remove(COUNT_COL_NAME)

    f = F.explode(F.arrays_zip(KMER_COL_NAME, COUNT_COL_NAME))
    kmerDf = (
        kmerDf
        .withColumn('tmp', f)
        .select(
            *gCols,
            F.col('tmp.kmer').alias(KMER_COL_NAME),
            F.col('tmp.count').alias(COUNT_COL_NAME)
        )
    )
    return kmerDf

def toDict(kmerDf):
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    kmerDf = toList(kmerDf, isSorted=False)

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

    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    if (isSorted):
        schema  = getSchema(countsAsInt=False)
        kmerDf = (
            kmerDf
            .groupby(KMERDF_SEQINFO_COL_NAMES)
            .applyInPandas(_sortCounts, schema=getSchema())
        )

    else:
        kmerDf = (
            kmerDf
            .groupby(KMERDF_SEQINFO_COL_NAMES)
            .agg(
                F.collect_list(KMER_COL_NAME).alias(KMER_COL_NAME),
                F.collect_list(COUNT_COL_NAME).alias(COUNT_COL_NAME)
            )
        )

    return kmerDf

def toVector(kmerDf, toDense=False):
    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    kmerDf = insertKmerIdxCol(kmerDf)
    kmerDf = kmerDf.withColumnRenamed(KMER_COL_NAME, 'kmerStr')

    ## Get the size of the vector
    vSize = kmerDf.select(F.max('kmerIdx')).collect()[0][0] + 1

    ## Create a dictionary containing (Kmer, Count) pairs
    kmerDf = kmerDf.withColumnRenamed('kmerIdx', KMER_COL_NAME)
    kmerDf = toDict(kmerDf)

    ## Convert the counts into SparseVectors
    f = lambda x: Vectors.sparse(vSize, x)
    f = F.udf(f, VectorUDT())
    kmerDf = kmerDf.withColumn(KMER_COL_NAME, f(KMER_COL_NAME))

    ## If applicable, convert the counts into DenseVectors
    if (toDense):
        f = lambda x: Vectors.dense(x.toArray())
        f = F.udf(f, VectorUDT())
        kmerDf = kmerDf.withColumn(KMER_COL_NAME, f(KMER_COL_NAME))

    ## Vectorising loses the Kmer string information. We probably don't need
    ## them, but if we do, then we can convert the vector indices into Kmers.
    return kmerDf

def groupBy(kmerDf, *cols, func=F.sum, sep=' '):
    nParts = kmerDf.rdd.getNumPartitions()

    ## Sum the counts for each Kmer
    x = (
        kmerDf
        .groupby(*cols, KMER_COL_NAME)
        .agg(func(COUNT_COL_NAME).alias(COUNT_COL_NAME))
    )

    ## Sum the lengths of each sequence
    y = (
        kmerDf
        .select(*cols, *KMERDF_SEQINFO_COL_NAMES).distinct()
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
    def _f(df):
        ## Create a column containing the reverse complement of each sequence
        f = lambda x: '-'.join(sorted(x))
        df['revComp'] = df[KMER_COL_NAME].apply(seqToRevComp)
        df[KMER_COL_NAME] = df[[KMER_COL_NAME, 'revComp']].apply(f, axis=1)

        ## Sum up the counts of complementary K-mers, and double the counts
        ## for the total frequency of each K-mer.
        df = (df.groupby([*KMERDF_SEQINFO_COL_NAMES, KMER_COL_NAME])
            .sum(numeric_only=True) * 2)
        df = df.reset_index()
        return df

    kmerDf = kmerDf.select(KMERDF_COL_NAMES)
    kmerDf = (kmerDf.groupby(KMERDF_SEQINFO_COL_NAMES)
        .applyInPandas(_f, schema=kmerDf.schema))
    return kmerDf

#------------------- Private Classes & Functions ------------#

def _toProbabilities(kmerPdf):
    total = kmerPdf[COUNT_COL_NAME].sum()
    kmerPdf[COUNT_COL_NAME] = kmerPdf[COUNT_COL_NAME] / total
    kmerPdf[COUNT_COL_NAME] = kmerPdf[COUNT_COL_NAME]
    return kmerPdf

def _toL2Normalised(kmerPdf):
    counts = kmerPdf[COUNT_COL_NAME].values.reshape(1, -1)
    kmerPdf[COUNT_COL_NAME] = preprocessing.normalize(counts)[0]
    kmerPdf[COUNT_COL_NAME] = kmerPdf[COUNT_COL_NAME]
    return kmerPdf

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
