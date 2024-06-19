#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import math

# External imports
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

# Internal imports
from .... import kmer
from .common import *

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def getLHMax(kmerDf):
    ## Calculate LHMax, based on formulas reported in:
    ## * Luczak et al. (2018)
    ## * Sims et al. (2009)
    seqLenDf  = kmerDf.select(kmer.SEQID_COL_NAME, kmer.SEQLEN_COL_NAME).distinct()
    lengthRdd = seqLenDf.rdd.mapValues(lambda x: math.log(x, 4))

    ## Calculate the average K-mer length
    lengths   = lengthRdd.values().collect()
    avgLength = np.mean(lengths)
    return avgLength

def getAcf(kmerDf):
    ## Calculate the average number of common Kmers between sequences
    ## Based on the formulas reported in:
    ## * Zhang et al. (2017)
    ## * Pornputtapong et al. (2020)

    ## Find all non-self pairs
    nSeqs  = kmerDf.select(kmer.SEQID_COL_NAME).distinct().count()
    kmerDf = getPairs(kmerDf)
    kmerDf = kmerDf.filter(F.col('l.rId') < F.col('r.rId'))

    ## Calculate the average number of common Kmers between each non-self pair
    f = F.size(F.array_intersect('l.kmer', 'r.kmer')) / (F.lit(nSeqs - 1))
    kmerDf = kmerDf.select(f.alias('commonKmers'))
    acf    = kmerDf.groupby().sum().collect()[0][0]
    return acf

def getFck(kmerDf):
    ## Calculate the fraction of Kmers present in all sequences
    ## Based on the approach reported in:
    ## * Gardner et al. (2013)
    nSeqs  = kmerDf.select(kmer.SEQID_COL_NAME).distinct().count()
    kmerDf = kmerDf.groupby(KMER_COL_NAME).count()

    ## Calculate the number of Kmers present in all sequences
    cKmers = kmerDf.filter(F.col(kmer.COUNT_COL_NAME) == nSeqs).count()
    uKmers = kmerDf.filter(F.col(kmer.COUNT_COL_NAME) != nSeqs).count()
    fck    = cKmers / (cKmers + uKmers)
    return fck

def getFuk(kmerDf):
    ## Calculate the average fraction of unique Kmers of each sequences
    ## Based on the approach reported in:
    ## * Gardner et al. (2013)

    ## Count the total number of Kmers for each sequence
    seqTotals = (
        kmerDf
        .groupby(kmer.SEQID_COL_NAME)
        .agg(F.sum(kmer.COUNT_COL_NAME).alias('t'))
    )

    ## Count the number of unique Kmers in each sequence
    uniqKmers = (
        kmerDf
        .filter(F.col(kmer.COUNT_COL_NAME) == 1)
        .groupby(kmer.SEQID_COL_NAME).agg(f.alias(kmer.COUNT_COL_NAME))
    )

    ## Calculate the fraction of unique Kmers in each sequence
    f = F.col(kmer.COUNT_COL_NAME) / F.col('t')
    kmerDf = (
        seqTotals
        .join(uniqKmers, on=kmer.SEQID_COL_NAME, how='left')
        .select(f.alias('uniqueKmers'))
    )

    ## Calculate the average number of unique Kmers across all sequences
    fuk = kmerDf.groupby().mean().collect()[0][0]
    return fuk

def getMinMaxRatio(kmerDf):
    ## Calculate the ratio between the longest and shortest sequence
    maxLen = kmerDf.select(F.max(kmer.SEQLEN_COL_NAME)).collect()[0][0]
    minLen = kmerDf.select(F.min(kmer.SEQLEN_COL_NAME)).collect()[0][0]
    print("Max. sequence length\t{}".format(maxLen))
    print("Min. sequence length\t{}".format(minLen))

    lenRatio = maxLen / minLen
    return lenRatio

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
