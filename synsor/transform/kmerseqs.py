#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import functools

# External imports
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports
from ..schema import *
from .. import util

#------------------- Constants ------------------------------#

#------------------- Classes & Functions --------------------#

def toInt(kmerDf):
    kmerDf = insertIntCol(kmerDf)
    kmerDf = kmerDf.drop(KMER_COL_NAME).withColumnRenamed('kmerIdx', KMER_COL_NAME)
    return kmerDf

def appendRevComp(kmerDf):
    f = F.pandas_udf(lambda x: x.apply(_appendRevComp), T.StringType())
    kmerDf = kmerDf.withColumn(KMER_COL_NAME, f(KMER_COL_NAME))
    return kmerDf

def insertIntCol(kmerDf):
    f = F.pandas_udf(lambda x: x.apply(_toInt), T.IntegerType())
    kmerDf = kmerDf.withColumn('kmerIdx', f(KMER_COL_NAME))
    return kmerDf

def _appendRevComp(s):
    s = sorted([s, util.seq.seqToRevComp(s)])
    s = '-'.join(s)
    return s

def _toInt(s):
    ## Taken from
    ## https://medium.com/computational-biology/vectorization-of-dna-sequences-2acac972b2ee
    kLen = len(s)
    s = (int(format(ord(c), 'b')) for c in s)
    s = map(lambda x: x >> 1 & 3, s)
    s = functools.reduce(lambda x, y: ((x << 2) & (4 ** kLen - 1)) + y, s)
    return s

#------------------- Main -----------------------------------#

