#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# # Standard library imports
import os
import time

# External imports
import typer
from typing import List, Optional
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Internal imports
from src.jobs import calculate_kmer_frequencies
from src.jobs import model_kmer_frequencies

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def main(
    i: List[str] = typer.Option([], help='FASTA/FASTQ files (.fa/.fq)'),
    f: List[str] = typer.Option(..., help='Directory containing K-mer frequencies (.avro)'),
    m: str = typer.Option(..., help='Predictive model (KerasClassifier)'),
    o: str = typer.Option(..., help='Output file (.tsv)'),
    t: int = typer.Option(8, help='Number of partitions')):

    """
    Predict the engineering status of DNA sequences
    """

    ## Calculate K-mer frequencies if they haven't been computed.
    if (len(i) != 0):
        ## Check that the FASTA files exist
        if (not _validFastaFiles(i)):
            raise Exception("Invalid files.")

        ## Check that there's only 1 path to write Kmer frequencies
        if (len(f) != 1):
            raise Exception("Must have exactly 1 path to write Kmer frequencies.")

        print("Calculating K-mer frequencies")
        with SparkSession.builder.config(conf=_getSparkConf(t)).getOrCreate() as spark:
            calculate_kmer_frequencies.main(i, 1, f[0])

    ## Check that the AVRO FREQ directories exist
    if (not _validAvroDirs(f)):
        raise Exception("Invalid directories.")

    print("Predicting sequence class")
    model_kmer_frequencies.main(f, m, o)

#------------------- Private Classes & Functions ------------#

def _getSparkConf(t):
    conf = (
        SparkConf()
        .set("spark.jars", "resources/spark-avro_2.12-3.3.1.jar")
        .set("spark.driver.maxResultSize", "0")
        .set("spark.executor.heartbeatInterval", "60s")
        .set("spark.sql.autoBroadcastJoinThreshold", "-1")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.ui.showConsoleProgress", "true")
        .set("spark.network.timeout", "600s")
        .set("spark.default.parallelism", t)
    )
    return conf

def _validFastaFiles(files):
    b = [all([os.path.isfile(f), os.path.exists(f)]) for f in files]
    return all(b)

def _validAvroDirs(dirs):
    b = [all([os.path.isdir(d), os.path.exists(d)]) for d in dirs]
    return all(b)

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    typer.run(main)

#------------------------------------------------------------------------------
