#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
from pathlib import Path

# External imports
import pytest
import pyspark.sql.types as T
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession

# Internal imports
from synsor import io
from synsor import schema
from synsor import transform

#------------------- Constants ------------------------------#

TEST_DATA_FASTA        = 'test/data/seqs.fna'
TEST_DATA_2MER_PARQUET = 'test/data/freqs_parquet_2mer'
TEST_DATA_3MER_PARQUET = 'test/data/freqs_parquet_3mer'

#------------------- Classes & Functions --------------------#

def test_kmercounts_toDict(spark, evenKmerDf):
    df = transform.kmercounts.toDict(evenKmerDf)
    nSeqs = df.select(schema.SEQID_COL_NAME).distinct().count()
    assert nSeqs == 25
    assert isinstance(df.schema[schema.KMER_COL_NAME].dataType, T.MapType)

def test_kmercounts_toList(spark, evenKmerDf):
    df = transform.kmercounts.toList(evenKmerDf)
    nSeqs = df.select(schema.SEQID_COL_NAME).distinct().count()
    assert nSeqs == 25
    assert isinstance(df.schema[schema.KMER_COL_NAME].dataType, T.ArrayType)
    assert isinstance(df.schema[schema.COUNT_COL_NAME].dataType, T.ArrayType)

    df = transform.kmercounts.toList(evenKmerDf, isSorted=False)
    nSeqs = df.select(schema.SEQID_COL_NAME).distinct().count()
    assert nSeqs == 25

def test_kmercounts_toVector_even(spark, evenKmerDf):
    df = transform.kmercounts.toVector(evenKmerDf)
    nSeqs = df.select(schema.SEQID_COL_NAME).distinct().count()
    vDim  = df.select(schema.KMER_COL_NAME).collect()[0][0].toArray().shape
    assert isinstance(df.schema[schema.KMER_COL_NAME].dataType, VectorUDT)
    assert nSeqs == 25
    assert vDim == (16, )

    df = transform.kmercounts.groupRevComp(evenKmerDf)
    df = transform.kmercounts.toVector(df)
    nSeqs = df.select(schema.SEQID_COL_NAME).distinct().count()
    vDim  = df.select(schema.KMER_COL_NAME).collect()[0][0].toArray().shape
    assert vDim == (10, )

    df = transform.kmercounts.groupRevComp(evenKmerDf)
    df = df.filter(df.kmer == 'AA-TT')
    df = transform.kmercounts.toVector(df)
    vDim = df.select(schema.KMER_COL_NAME).collect()[0][0].toArray().shape
    assert vDim == (1, )

def test_kmercounts_toVector_odd(spark, oddKmerDf):
    df = transform.kmercounts.toVector(oddKmerDf)
    nSeqs = df.select(schema.SEQID_COL_NAME).distinct().count()
    vDim  = df.select(schema.KMER_COL_NAME).collect()[0][0].toArray().shape
    assert vDim == (64, )

    df = transform.kmercounts.groupRevComp(oddKmerDf)
    df = transform.kmercounts.toVector(df)
    vDim = df.select(schema.KMER_COL_NAME).collect()[0][0].toArray().shape
    assert vDim == (32, )

def test_kmercounts_groupRevComp_even(spark, evenKmerDf):
    df = transform.kmercounts.groupRevComp(evenKmerDf)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    c1 = df.filter(df.kmer == 'AA-TT').sort(schema.SEQID_COL_NAME)
    c1 = c1.select(schema.COUNT_COL_NAME).collect()[0][0]
    c2 = df.filter(df.kmer == 'AT-AT').sort(schema.SEQID_COL_NAME)
    c2 = c2.select(schema.COUNT_COL_NAME).collect()[0][0]
    assert nKmers == 10
    assert c1 == 12076
    assert c2 == 4594
    assert evenKmerDf.schema.names == df.schema.names

def test_kmercounts_groupRevComp_odd(spark, oddKmerDf):
    df = transform.kmercounts.groupRevComp(oddKmerDf)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    c3 = df.filter(df.kmer == 'AAA-TTT').sort(schema.SEQID_COL_NAME)
    c3 = c3.select(schema.COUNT_COL_NAME).collect()[0][0]
    assert nKmers == 32
    assert c3 == 3778
    assert oddKmerDf.schema.names == df.schema.names

def test_kmerseqs_appendRevComp_even(spark, evenKmerDf):
    df = transform.kmerseqs.appendRevComp(evenKmerDf)
    c1 = (
        df.filter(df.kmer == 'AA-TT')
        .groupby(schema.SEQID_COL_NAME, schema.KMER_COL_NAME).count()
        .sort(schema.SEQID_COL_NAME, schema.KMER_COL_NAME)
        .select(schema.COUNT_COL_NAME).collect()[0][0]
    )
    c2 = (
        df.filter(df.kmer == 'AT-AT')
        .groupby(schema.SEQID_COL_NAME, schema.KMER_COL_NAME).count()
        .sort(schema.SEQID_COL_NAME, schema.KMER_COL_NAME)
        .select(schema.COUNT_COL_NAME).collect()[0][0]
    )
    assert c1 == 2
    assert c2 == 1
    assert evenKmerDf.count() == df.count()
    assert evenKmerDf.schema.names == df.schema.names

def test_kmerseqs_appendRevComp_odd(spark, oddKmerDf):
    df = transform.kmerseqs.appendRevComp(oddKmerDf)
    c1 = (
        df.filter(df.kmer == 'AAA-TTT')
        .groupby(schema.SEQID_COL_NAME, schema.KMER_COL_NAME).count()
        .sort(schema.SEQID_COL_NAME, schema.KMER_COL_NAME)
        .select(schema.COUNT_COL_NAME).collect()[0][0]
    )
    c2 = (
        df.filter(df.kmer == 'ATA-TAT')
        .groupby(schema.SEQID_COL_NAME, schema.KMER_COL_NAME).count()
        .sort(schema.SEQID_COL_NAME, schema.KMER_COL_NAME)
        .select(schema.COUNT_COL_NAME).collect()[0][0]
    )
    assert c1 == 2
    assert c2 == 2
    assert oddKmerDf.count() == df.count()
    assert oddKmerDf.schema.names == df.schema.names

def test_kmerseqs_insertIntCol_even(spark, evenKmerDf):
    df = transform.kmerseqs.insertIntCol(evenKmerDf)
    df = df.filter(df.kmer == 'AA').sort(schema.SEQID_COL_NAME)
    c1 = df.select('kmerIdx').collect()[0][0]
    c2 = df.select(schema.COUNT_COL_NAME).collect()[0][0]
    assert c1 == 0
    assert c2 == 2832
    assert evenKmerDf.schema.names != df.schema.names

def test_kmerseqs_insertIntCol_odd(spark, oddKmerDf):
    df = transform.kmerseqs.insertIntCol(oddKmerDf)
    df = df.filter(df.kmer == 'AAA').sort(schema.SEQID_COL_NAME)
    c1 = df.select('kmerIdx').collect()[0][0]
    c2 = df.select(schema.COUNT_COL_NAME).collect()[0][0]
    assert c1 == 0
    assert c2 == 888
    assert oddKmerDf.schema.names != df.schema.names

def test_table_insertZeroCounts_even(spark, evenKmerDf):
    df = evenKmerDf.filter(evenKmerDf.kmer == 'AA')
    df = transform.table.insertZeroCounts(df)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    assert nKmers == 1

    df = evenKmerDf.filter(evenKmerDf.kmer == 'AA')
    df = transform.table.insertZeroCounts(df, possibleKmers=True)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    assert nKmers == 16

    df = transform.kmercounts.groupRevComp(evenKmerDf)
    df = df.filter(df.kmer == 'AA-TT')
    df = transform.table.insertZeroCounts(df, possibleKmers=True)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    assert nKmers == 10

def test_table_insertZeroCounts_even(spark, oddKmerDf):
    df = oddKmerDf.sample(0.01, seed=42)
    df = transform.table.insertZeroCounts(df)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    assert nKmers == 17

    df = oddKmerDf.filter(oddKmerDf.kmer == 'AAA')
    df = transform.table.insertZeroCounts(df, possibleKmers=True)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    assert nKmers == 64

    df = transform.kmercounts.groupRevComp(oddKmerDf)
    df = df.filter(df.kmer == 'AAA-TTT')
    df = transform.table.insertZeroCounts(df, possibleKmers=True)
    nKmers = df.select(schema.KMER_COL_NAME).distinct().count()
    assert nKmers == 32

@pytest.fixture
def evenKmerDf(spark):
    df = io.kmer.read(Path(TEST_DATA_2MER_PARQUET))
    return df

@pytest.fixture
def oddKmerDf(spark):
    df = io.kmer.read(Path(TEST_DATA_3MER_PARQUET))
    return df

#------------------- Main -----------------------------------#
