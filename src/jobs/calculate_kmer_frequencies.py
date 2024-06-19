#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports

# External imports
from pyspark.sql import SparkSession

# Internal imports
from ..shared import io
from ..shared import seq2kmer

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def main(iFiles, kmerLength, oFile, ignoreNs=True, countExp=False):
    ## Read data
    ss = SparkSession.getActiveSession()
    sc = ss.sparkContext
    seqRecRdd = sc.parallelize(iFiles, len(iFiles))
    seqRecRdd = seqRecRdd.flatMap(io.fastx.read)
    seqRecRdd = seqRecRdd.repartition(sc.defaultParallelism)
    ## We encounter a problem when there are too many sequences in
    ## a single file. This can be overcome by:
    ## * Spliting up the larger input file into smaller input files
    ## * Chunking the sequence records and repeated flatmap/repartitions

    ## Get a table containing the kmer counts across for each record
    kmerDf    = seq2kmer.getCounts(seqRecRdd, kmerLength,
        ignoreNs, countExp)

    ## Write the table to disk
    print("Writing output")
    io.kmer.write(oFile, kmerDf)

#------------------- Private Classes & Functions ------------#

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
