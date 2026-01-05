#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
from operator import add

# External imports
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports
from .util import seq
from .schema import *

#------------------- Constants ------------------------------#

#------------------- Classes & Functions --------------------#

def getCounts(rdd, kmerLength, ignoreNs, countExp):
    rdd = rdd.map(lambda x: (x.description, str(x.seq).upper()))
    if (countExp):
        print("Expected counts")
        kImerLength  = kmerLength - 1
        kIImerLength = kmerLength - 2

        kImerDf  = _getObservedCounts(rdd, kImerLength, ignoreNs)
        seqLenDf = _getSeqLengths(rdd)
        kImerDf  = _createKmerDf(kImerDf, seqLenDf)
        kImerDf  = (
            kImerDf.withColumnRenamed(KMER_COL_NAME, 'kImer')
            .withColumnRenamed(COUNT_COL_NAME, 'kImerCount')
        )

        kIImerDf = _getObservedCounts(rdd, kIImerLength, ignoreNs)
        seqLenDf = _getSeqLengths(rdd)
        kIImerDf = _createKmerDf(kIImerDf, seqLenDf)
        kIImerDf = (
            kIImerDf.withColumnRenamed(KMER_COL_NAME, 'kIImer')
            .withColumnRenamed(COUNT_COL_NAME, 'kIImerCount')
        )

        kmerDf = _getExpectedCounts(kImerDf, kIImerDf, kImerLength, kIImerLength)

    else:
        kmerDf   = _getObservedCounts(rdd, kmerLength, ignoreNs)
        seqLenDf = _getSeqLengths(rdd)
        kmerDf   = _createKmerDf(kmerDf, seqLenDf)
        kmerDf   = kmerDf.withColumn(COUNT_COL_NAME,
            F.col(COUNT_COL_NAME).cast(T.IntegerType()))

    return kmerDf

def _getObservedCounts(rdd, kmerLength, ignoreNs):
    def _getKmers(seq, kmerLength):
        obsTotal = _getTotalKmerCount(seq, kmerLength)
        return (seq[i:i + kmerLength] for i in range(0, obsTotal))

    def _getTotalKmerCount(seq, kmerLength):
        return len(seq) - kmerLength + 1

    ## (ID, Seq) => (ID, kmerSeq)
    ##             => ((ID, kmerSeq), 1)
    ##             => ((ID, kmerSeq), count)
    f = lambda x: _getKmers(x, kmerLength)
    kmerRdd = rdd.flatMapValues(f).map(lambda x: (x, 1))
    kmerRdd = kmerRdd.repartition(kmerRdd.getNumPartitions() * kmerLength)
    kmerRdd = kmerRdd.reduceByKey(add).map(lambda x: (*x[0], x[1]))
    kmerDf  = kmerRdd.toDF([SEQID_COL_NAME, KMER_COL_NAME, COUNT_COL_NAME])
    kmerDf  = kmerDf.repartition(kmerDf.rdd.getNumPartitions(), SEQID_COL_NAME)

    ## We only count Kmers that occur at least once. Kmers that
    ## do not occur (zero counts) are ignored (which will save us quite
    ## a bit of space), but need to be added when we're analysing the counts.
    if (ignoreNs):
        ## Ensure that we don't have Kmers containing ambiguous bases
        ## i.e., N's, R's, S's, etc...
        nStr = ''.join(seq.NUCLEOTIDES)
        p    = '^[' + nStr + ']+$'
        kmerDf = kmerDf.filter(kmerDf.kmer.rlike(p))

    return kmerDf

def _getSeqLengths(rdd):
    ## (ID, Seq) => (ID, len(Seq))
    seqLenRdd = rdd.mapValues(len)
    seqLenDf  = seqLenRdd.toDF([SEQID_COL_NAME, SEQLEN_COL_NAME])
    seqLenDf  = seqLenDf.repartition(seqLenDf.rdd.getNumPartitions(), SEQID_COL_NAME)
    return seqLenDf

def _createKmerDf(kmerDf, seqLenDf):
    kmerDf = (
        kmerDf.join(seqLenDf, on=SEQID_COL_NAME, how='left')
        .select(*KMERDF_COL_NAMES)
    )
    return kmerDf

def _getExpectedCounts(kImerDf, kIImerDf, kImerLength, kIImerLength):
    ## Join the tables
    cond   = (F.substring(kImerDf.kImer, 2, kIImerLength) == kIImerDf.kIImer)
    kmerDf = (
        kImerDf.join(kIImerDf, on=[SEQID_COL_NAME, SEQLEN_COL_NAME], how='full')
        .filter(cond)
        .withColumnRenamed('kImer', 'preKmer')
        .withColumnRenamed('kImerCount', 'preKmerCount')
        .withColumnRenamed('kIImer', 'inKmer')
        .withColumnRenamed('kIImerCount', 'inKmerCount')
    )

    cond   = (F.substring(kImerDf.kImer, 1, kIImerLength) == kmerDf.inKmer)
    kmerDf = (
        kmerDf.join(kImerDf, on=[SEQID_COL_NAME, SEQLEN_COL_NAME], how='full')
        .filter(cond)
        .withColumnRenamed('kImer', 'suffKmer')
        .withColumnRenamed('kImerCount', 'suffKmerCount')
    )

    ## Calculate the Expected counts
    f = (F.col('preKmerCount') * F.col('suffKmerCount')) / F.col('inKmerCount')
    g = (
        F.concat(F.substring(kmerDf.preKmer, 1, 1),
            kmerDf.inKmer,
            F.substring(kmerDf.suffKmer, kImerLength, 1)
        )
    )
    kmerDf = kmerDf.withColumn(COUNT_COL_NAME, f)
    kmerDf = kmerDf.withColumn(KMER_COL_NAME, g)

    ## Return the columns we want
    colNames = [
        SEQID_COL_NAME, SEQLEN_COL_NAME,
        KMER_COL_NAME, COUNT_COL_NAME
    ]
    return kmerDf.select(colNames)

#------------------- Main -----------------------------------#

