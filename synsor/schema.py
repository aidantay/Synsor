#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports

#------------------- Constants ------------------------------#

## Column names
SEQID_COL_NAME  = 'seqId'
SEQLEN_COL_NAME = 'seqLen'
KMER_COL_NAME   = 'kmer'
COUNT_COL_NAME  = 'count'

KMERDF_SEQINFO_COL_NAMES   = [SEQID_COL_NAME, SEQLEN_COL_NAME]
KMERDF_KMERCOUNT_COL_NAMES = [KMER_COL_NAME, COUNT_COL_NAME]
KMERDF_COL_NAMES           = [*KMERDF_SEQINFO_COL_NAMES, *KMERDF_KMERCOUNT_COL_NAMES]

#------------------- Classes & Functions --------------------#

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

#------------------- Main -----------------------------------#

