#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import itertools
import functools

# External imports
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports
from ..common import *

#------------------- Constants ------------------------------#

## Column names
SEQID_COL_NAME  = 'seqId'
SEQLEN_COL_NAME = 'seqLen'
KMER_COL_NAME   = 'kmer'
COUNT_COL_NAME  = 'count'

KMERDF_SEQINFO_COL_NAMES   = [SEQID_COL_NAME, SEQLEN_COL_NAME]
KMERDF_KMERCOUNT_COL_NAMES = [KMER_COL_NAME, COUNT_COL_NAME]
KMERDF_COL_NAMES           = [*KMERDF_SEQINFO_COL_NAMES, *KMERDF_KMERCOUNT_COL_NAMES]

#------------------- Public Classes & Functions -------------#

def getSchema(countsAsInt=True):
    colTypes = [
        T.StringType(),
        T.LongType(),
        T.ArrayType(T.StringType()),
        T.ArrayType(T.IntegerType()) if countsAsInt else T.ArrayType(T.FloatType())
    ]
    cols = [T.StructField(c, t) for c, t in zip(KMERDF_COL_NAMES, colTypes)]
    schema = T.StructType(cols)
    return schema

def getExpectedSequences(kmerLength):

    """
    Description:
        Generates a list of all possible Kmer sequences of length K.

    Args:
        kmerLength (int):
            Length of Kmers. Must be a positive integer.

    Returns:
        kmerSeqs (generator):
            List of all possible Kmer sequences.

    Raises:
        ValueError:
            If kmerLength is not a positive integer.
    """

    f = itertools.product(NUCLEOTIDES, repeat=kmerLength)
    return (''.join(c) for c in f)

def getExpectedTotal(seq_or_kLen):
    ## Check whether we're dealing with a Kmer sequence
    ## or a Kmer length
    if (isinstance(seq_or_kLen, str)):
        seq = seq_or_kLen
        ## Check whether the Kmer sequence is paired with its reverse
        ## complement (i.e., FWD-REV)
        if ('-' in seq):
            kmer = seq.split('-')[0]
            kLen = len(kmer)
            ## Check whether the length of the Kmer sequence is even or odd
            if (kLen % 2 == 0):
                ## Even K-mer lengths can generate palindromes which
                ## changes the total number of possible K-mers.
                ## Based on formulas reported in:
                ## * Apostolou-Karampelis et al. (2019)
                x = (2 * kLen) - 1
                y = (kLen - 1)
                t = (2 ** x) + (2 ** y)
                return t

            else:
                t = (4 ** kLen) / 2
                return int(t)

        else:
            kLen = len(seq)
            return (4 ** kLen)

    else:
        kLen = seq_or_kLen
        return (4 ** kLen)

def seqToInt(s):
    ## Taken from
    ## https://medium.com/computational-biology/vectorization-of-dna-sequences-2acac972b2ee
    kLen = len(s)
    s = (int(format(ord(c), 'b')) for c in s)
    s = map(lambda x: x >> 1 & 3, s)
    s = functools.reduce(lambda x, y: ((x << 2) & (4 ** kLen - 1)) + y, s)
    return s

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
