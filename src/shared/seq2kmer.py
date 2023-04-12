#!/bin/python

#------------------- Description & Notes --------------------#

'''
Kmer frequencies only need to be calcuated in either the forward, or
the reverse direction since they basically contain the same information.
Specifically, for odd K-mers,  Frequencies(forward) == Frequencies(reverse).
* Total N(AAA) = N(AAA, Forward) + N(AAA, Reverse)
               = N(AAA, Forward) + N(TTT, Reverse)

However, for even K-mers, Frequencies(forward) != Frequencies(reverse) because
even K-mers can generate palindromes (Odd K-mers never generate palindromes)
* Total N(AT) = N(AT, Forward) + N(AT, Reverse)
              = N(AT, Forward) + N(AT, Forward)
              = N(AT, Forward) * 2
* Total N(AGCT) = N(AGCT, Forward) + N(AGCT, Reverse)
*               = N(AGCT, Forward) + N(AGCT, Forward)
*               = N(AGCT, Forward) * 2
Thus, for these K-mers, the frequencies are doubled instead of additive.

The above suggests that only odd K-mers should be considered to avoid the
frequency biases due to different strands.
'''

#------------------- Dependencies ---------------------------#

# Standard library imports
from operator import add

# External imports
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports
from . import NUCLEOTIDES
from . import kmer

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

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
            kImerDf
            .withColumnRenamed(kmer.KMER_COL_NAME, 'kImer')
            .withColumnRenamed(kmer.COUNT_COL_NAME, 'kImerCount')
        )

        kIImerDf = _getObservedCounts(rdd, kIImerLength, ignoreNs)
        seqLenDf = _getSeqLengths(rdd)
        kIImerDf = _createKmerDf(kIImerDf, seqLenDf)
        kIImerDf = (
            kIImerDf
            .withColumnRenamed(kmer.KMER_COL_NAME, 'kIImer')
            .withColumnRenamed(kmer.COUNT_COL_NAME, 'kIImerCount')
        )

        kmerDf = _getExpectedCounts(kImerDf, kIImerDf, kImerLength, kIImerLength)

    else:
        print("Observed counts")
        ## **********
        ## *** Not sure if this is the "best" approach for split counts.
        ## *** But seems to run the fastest!
        ## ***
        ## *** Using RDD ReduceByKey once to minimise shuffling can encounter
        ## *** a few errors (not sure why).
        ## ***
        ## *** I've tried:
        ## ***  * Using fewer lambda functions to reduce double serialisation.
        ## ***  * Repartitioning before reducing. However, this doesn't seem to
        ## ***    result in any major improvements since we're shuffling data twice.
        ## **********
        kmerDf   = _getObservedCounts(rdd, kmerLength, ignoreNs)
        seqLenDf = _getSeqLengths(rdd)
        kmerDf   = _createKmerDf(kmerDf, seqLenDf)
        kmerDf   = kmerDf.withColumn(kmer.COUNT_COL_NAME,
            F.col(kmer.COUNT_COL_NAME).cast(T.IntegerType()))

    return kmerDf

#------------------- Private Classes & Functions ------------#

def _getObservedCounts(rdd, kmerLength, ignoreNs):
    ## (ID, Seq) => (ID, kmerSeq)
    ##             => ((ID, kmerSeq), 1)
    ##             => ((ID, kmerSeq), count)
    f = lambda x: _getObservedSequences(x, kmerLength)
    kmerRdd = rdd.flatMapValues(f).map(lambda x: (x, 1))
    kmerRdd = kmerRdd.repartition(kmerRdd.getNumPartitions() * kmerLength)
    kmerRdd = kmerRdd.reduceByKey(add).map(lambda x: (*x[0], x[1]))
    kmerDf  = kmerRdd.toDF([kmer.SEQID_COL_NAME, kmer.KMER_COL_NAME, kmer.COUNT_COL_NAME])
    kmerDf  = kmerDf.repartition(kmerDf.rdd.getNumPartitions(), kmer.SEQID_COL_NAME)

    ## We only count Kmers that occur at least once. Kmers that
    ## do not occur (zero counts) are ignored (which will save us quite
    ## a bit of space), but need to be added when we're analysing the counts.
    if (ignoreNs):
        ## Ensure that we don't have Kmers containing ambiguous bases
        ## i.e., N's, R's, S's, etc...
        nStr = ''.join(NUCLEOTIDES)
        p    = '^[' + nStr + ']+$'
        kmerDf = kmerDf.filter(kmerDf.kmer.rlike(p))

    return kmerDf

def _getSeqLengths(rdd):
    ## (ID, Seq) => (ID, len(Seq))
    seqLenRdd = rdd.mapValues(len)
    seqLenDf  = seqLenRdd.toDF([kmer.SEQID_COL_NAME, kmer.SEQLEN_COL_NAME])
    seqLenDf  = seqLenDf.repartition(seqLenDf.rdd.getNumPartitions(), kmer.SEQID_COL_NAME)
    return seqLenDf

def _createKmerDf(kmerDf, seqLenDf):
    kmerDf = (
        kmerDf
        .join(seqLenDf, on=kmer.SEQID_COL_NAME, how='left')
        .select(*kmer.KMERDF_COL_NAMES)
    )
    return kmerDf

def _getObservedSequences(seq, kmerLength):

    """
    Description:
        Generates a list of Kmer sequences of length K in a sequence.

    Args:
        seq (str):
            Sequence to be examined.

        kmerLength (int):
            Length of Kmers. Must be a positive integer.

    Returns:
        kmerSeqs (list):
            List of str containing the Kmer sequences of length K
            in the sequence record
    """

    obsTotal = _getObservedTotal(seq, kmerLength)
    return (seq[i:i + kmerLength] for i in range(0, obsTotal))

def _getObservedTotal(seq, kmerLength):
    return len(seq) - kmerLength + 1

def _getExpectedCounts(kImerDf, kIImerDf, kImerLength, kIImerLength):
    ## Join the tables
    cond   = (F.substring(kImerDf.kImer, 2, kIImerLength) == kIImerDf.kIImer)
    kmerDf = (
        kImerDf
        .join(kIImerDf, on=[kmer.SEQID_COL_NAME, kmer.SEQLEN_COL_NAME], how='full')
        .filter(cond)
        .withColumnRenamed('kImer', 'preKmer')
        .withColumnRenamed('kImerCount', 'preKmerCount')
        .withColumnRenamed('kIImer', 'inKmer')
        .withColumnRenamed('kIImerCount', 'inKmerCount')
    )

    cond   = (F.substring(kImerDf.kImer, 1, kIImerLength) == kmerDf.inKmer)
    kmerDf = (
        kmerDf
        .join(kImerDf, on=[kmer.SEQID_COL_NAME, kmer.SEQLEN_COL_NAME], how='full')
        .filter(cond)
        .withColumnRenamed('kImer', 'suffKmer')
        .withColumnRenamed('kImerCount', 'suffKmerCount')
    )

    ## Calculate the Expected counts
    f = (F.col('preKmerCount') * F.col('suffKmerCount')) / F.col('inKmerCount')
    g = F.concat(
        F.substring(kmerDf.preKmer, 1, 1),
        kmerDf.inKmer,
        F.substring(kmerDf.suffKmer, kImerLength, 1)
    )
    kmerDf = kmerDf.withColumn(kmer.COUNT_COL_NAME, f)
    kmerDf = kmerDf.withColumn(kmer.KMER_COL_NAME, g)

    ## Return the columns we want
    colNames = [
        kmer.SEQID_COL_NAME, kmer.SEQLEN_COL_NAME,
        kmer.KMER_COL_NAME, kmer.COUNT_COL_NAME
    ]
    return kmerDf.select(colNames)

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
