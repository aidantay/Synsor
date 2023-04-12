#!/bin/python

#------------------- Description & Notes --------------------#

'''
Description:
    Given a list of FASTA/FASTQ or SAM/BAM files, output (to file) the
    frequencies for each oligonucleotide sequence of length K in a sequence/s.
    Frequencies are calculated for each sequence record.

Args:
    fastXFiles (filepath):
        List containing the filepath of each file. Files can be
        compressed (.gz) and should contain at least one
        sequence record (FASTA/FASTQ or SAM/BAM). Our processing limit
        seems to be around 3.5 million (3,500,000) sequence records.

    kmerLength (int):
        Length of oligonucleotide sequences. Must be a positive integer.
        Ideally, this should be <= 13 since the total number of possible
        oligonucleotide sequences exponentially increases (4^K).
            * 4^13 Kmers =    67,108,864  ## Possible
            * 4^14 Kmers =   268,435,456  ## Sometimes possible
            * 4^15 Kmers = 1,073,741,824  ## Probably not possible

Returns:
    oFile (dir):
        Directory containing a list of files. Each file is bzip2
        compressed in Avro format and contains the frequencies for each
        oligonucleotide sequence of length K in a sequence/s.

Most issues can be fixed with one of the following:
* Increasing the number of partitions (i.e., spark.default.parallelism)
* Increasing the amount of memory (i.e., spark.*.memory)
* Splitting larger files into several smaller files
'''

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
