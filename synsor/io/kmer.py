#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import functools
import logging
import math
from pathlib import Path

# External imports
import pandas as pd
import dask.bag as db
import dask.dataframe as dd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Internal imports
from .. import transform
from ..schema import *

#------------------- Constants ------------------------------#

logger = logging.getLogger(__name__)

#------------------- Classes & Functions --------------------#

def read(*dirs, sample=1.0, verbose=False):
    ss = SparkSession.getActiveSession()
    if (ss is None):
        logger.info("No active Spark session. Reading with Dask")
        kmerDf = [_readWithDask(d, sample) for d in dirs]
        kmerDf = dd.concat(kmerDf)

    else:
        kmerDf = [_readWithSpark(d, sample, verbose) for d in dirs]
        kmerDf = functools.reduce(lambda x,y: x.union(y), kmerDf)

        ## Repartition the dataframe to maximise the number of partitions used
        ## This will (hopefully) speed things up
        kmerDf = kmerDf.repartitionByRange(ss.sparkContext.defaultParallelism,
            KMERDF_SEQINFO_COL_NAMES)
        _printPartitioningInfo(kmerDf, verbose)

    return kmerDf

def write(file, kmerDf, fmt='parquet'):
    ss = SparkSession.getActiveSession()
    if (ss is None):
        raise EnvironmentError("Must have an active Spark session")

    if (fmt == 'tsv'):
        ## Save k-mer sequences separately because these will be
        ## lost after vectorisation
        kmerStrToIdx = transform.kmerseqs.insertIntCol(kmerDf)
        kmerStrToIdx = kmerStrToIdx.select(KMER_COL_NAME, 'kmerIdx').distinct()

        ## Calculate the expected number of partitions/files based on the
        ## data. This should be proportional to the number of sequence records
        kmerDf = transform.kmercounts.toVector(kmerDf)
        nSeqs  = kmerDf.select(SEQID_COL_NAME).distinct().count()
        nParts = 1 + math.floor(math.log(nSeqs, 10))
        kmerDf = kmerDf.coalesce(nParts)

        ## Try to load everything into memory
        (kmerId, kmerCount) = transform.table.toDriver(kmerDf)
        kmerCount = pd.DataFrame(kmerCount.toarray())
        kmerCount = kmerCount.loc[:, (kmerCount != 0).any(axis=0)]
        kmerDf = pd.concat([kmerId, kmerCount], axis=1)

        ## Load the table back to Spark
        kmerDf = ss.createDataFrame(kmerDf)
        kmerDf = kmerDf.coalesce(nParts)
        o = file / 'counts'
        kmerDf.write.csv(o, mode='overwrite', sep='\t', header=True, compression='gzip')

        ## Write the table to disk
        o = file / 'kmers'
        kmerStrToIdx = kmerStrToIdx.coalesce(nParts)
        kmerStrToIdx.write.csv(str(o), mode='overwrite', sep='\t', header=True, compression='gzip')

    else:
        ## Aggregate the Kmers. This makes reading/writing a bit easier.
        kmerDf = transform.kmercounts.toList(kmerDf, isSorted=False)
        kmerDf = kmerDf.withColumn(SEQLEN_COL_NAME,
            F.col(SEQLEN_COL_NAME).cast(T.LongType()))

        ## Calculate the expected number of partitions/files based on the
        ## data. This should be proportional to the number of sequence records
        nSeqs  = kmerDf.select(SEQID_COL_NAME).distinct().count()
        nParts = 1 + math.floor(math.log(nSeqs, 10))
        kmerDf = kmerDf.coalesce(nParts)

        if (fmt == 'parquet'):
            kmerDf.write.parquet(str(file), mode='overwrite', compression="snappy")

        elif (fmt == 'avro'):
            ## This should be more appropriate and more performant than parquet
            ## For more info, see:
            ## * https://luminousmen.com/post/big-data-file-formats
            kmerDf.write.format('avro').save(str(file), mode='overwrite', compression='bzip2')

        else:
            raise NotImplementedError('Invalid format.')

def _readWithDask(d, sample):
    ## Collect files
    p = list(d.glob('*.parquet'))
    a = list(d.glob('*.avro'))

    if (len(p) > 1 and len(a) == 0):
        ## Read parquet files
        kmerDf = dd.read_parquet(str(d))

    elif (len(a) > 1 and len(p) == 0):
        ## Read avro files
        kmerDf = db.read_avro(a).to_dataframe()

    else:
        raise EnvironmentError("Directory contains multiple formats.")

    ## Repartition the dataframe to maximise the number of partitions used
    ## This will (hopefully) speed things up
    kmerDf = kmerDf.repartition(partition_size='10MB')

    if (sample != 1.0):
        kmerDf = kmerDf.sample(frac=sample)

    ## Explode Kmers
    kmerDf[['kmer', 'count']] = kmerDf[['kmer', 'count']].astype(object)
    kmerDf = kmerDf.explode(['kmer', 'count'])
    return kmerDf

def _readWithSpark(d, sample, verbose):
    ss = SparkSession.getActiveSession()

    ## Collect files
    p = list(d.glob('*.parquet'))
    a = list(d.glob('*.avro'))
    if (len(p) > 1 and len(a) == 0):
        ## Read parquet files
        kmerDf = ss.read.parquet(str(d))

    elif (len(a) > 1 and len(p) == 0):
        ## Read avro files
        kmerDf = ss.read.format('avro').load(str(d), schema=getSchema())

    else:
        raise EnvironmentError("Directory contains multiple formats.")

    ## Repartition the dataframe to maximise the number of partitions used
    ## This will (hopefully) speed things up
    kmerDf = kmerDf.repartitionByRange(ss.sparkContext.defaultParallelism,
        KMERDF_SEQINFO_COL_NAMES)

    ## If applicable, sample the data (reduces the amount of data analysed)
    if (sample != 1.0):
        kmerDf = kmerDf.sample(fraction=sample)

    ## Explode the Kmers
    _printPartitioningInfo(kmerDf, verbose)
    kmerDf = transform.kmercounts.explode(kmerDf)
    return kmerDf

def _printPartitioningInfo(kmerDf, verbose):
    nParts = kmerDf.rdd.getNumPartitions()
    nSeqsPerPart = len(
        kmerDf.select(SEQID_COL_NAME).rdd
        .glom().first()
    )

    if (verbose):
        ## Print some partitioning info (for optimising scalability)
        logger.info(f"Total Num. Partitions\t{nParts}")
        logger.info(f"Approx Num. Seqs Per Partition\t{nSeqsPerPart}")

#------------------- Main -----------------------------------#

