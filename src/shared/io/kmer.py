#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import math

# External imports
import dask.bag as db
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports
from .. import kmer
from ..kmer import transform

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def read(*dirs, sample=1.0):
    ss = SparkSession.getActiveSession()
    if (ss is None):
        print("No active Spark session. Reading with Dask")
        kmerDf = _readWithDask(*dirs, sample=sample)

    else:
        kmerDf = _readWithSpark(*dirs, sample=sample)

    return kmerDf

def write(filepath, kmerDf):
    ss = SparkSession.getActiveSession()
    if (ss is None):
        raise EnvironmentError("Must have an active Spark session")

    ## Aggregate the Kmers. This makes reading/writing a bit easier.
    kmerDf = transform.counts.toList(kmerDf, isSorted=False)
    kmerDf = kmerDf.withColumn(kmer.SEQLEN_COL_NAME,
        F.col(kmer.SEQLEN_COL_NAME).cast(T.LongType()))

    ## Calculate the expected number of partitions/files based on the
    ## data. This should be proportional to the number of sequence records
    nSeqs  = kmerDf.select(kmer.SEQID_COL_NAME).distinct().count()
    nParts = 1 + math.floor(math.log(nSeqs, 10))
    kmerDf = kmerDf.coalesce(nParts)

    ## Write the table to file in AVRO format
    ## This should be more appropriate and more performant than other formats.
    ## For more info, see:
    ## * https://luminousmen.com/post/big-data-file-formats
    kmerDf.write.format('avro').save(filepath, mode='overwrite', compression='bzip2')

#------------------- Private Classes & Functions ------------#

def _readWithDask(*dirs, sample):
    ## Collect files and read AVRO tables
    dirpaths = [d + '/*.avro' for d in dirs]
    b = db.read_avro(dirpaths)
    kmerDf = b.to_dataframe()
    print(kmerDf)

    ## Repartition the dataframe to maximise the number of partitions used
    ## This will (hopefully) speed things up
    kmerDf = kmerDf.repartition(partition_size='10MB')
    print(kmerDf)

    if (sample != 1.0):
        kmerDf = kmerDf.sample(frac=sample)

    ## Explode Kmers
    kmerDf = kmerDf.explode(['kmer', 'count'])
    return kmerDf

def _readWithSpark(*dirs, sample):
    ## Collect files and read AVRO tables
    dirpaths = [d + '/*.avro' for d in dirs]
    ss = SparkSession.getActiveSession()
    kmerDf = ss.read.format('avro').load(dirpaths, schema=kmer.getSchema())

    ## Repartition the dataframe to maximise the number of partitions used
    ## This will (hopefully) speed things up
    kmerDf = kmerDf.repartitionByRange(ss.sparkContext.defaultParallelism,
        kmer.KMERDF_SEQINFO_COL_NAMES)

    ## If applicable, sample the data (reduces the amount of data analysed)
    if (sample != 1.0):
        kmerDf = kmerDf.sample(fraction=sample)

    ## Explode the Kmers
    _printPartitioningInfo(kmerDf)
    kmerDf = transform.counts.explode(kmerDf)
    return kmerDf

def _printPartitioningInfo(kmerDf):
    nParts = kmerDf.rdd.getNumPartitions()
    nSeqsPerPart = len(
        kmerDf.select(kmer.SEQID_COL_NAME).rdd
        .glom().first()
    )

    ## Print some partitioning info (for optimising scalability)
    print("Total Num. Partitions\t{}".format(nParts))
    print("Approx Num. Seqs Per Partition\t{}".format(nSeqsPerPart))

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    main()

#------------------------------------------------------------------------------
