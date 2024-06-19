#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# # Standard library imports
import argparse
import os
import time
import sys

# External imports
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Internal imports
from src.jobs import calculate_kmer_frequencies
from src.jobs import predict_engineered_sequences
from src.jobs.aux import explore_kmer_frequencies
from src.jobs.aux import select_models
from src.jobs.aux import build_models
from src.jobs.aux import train_test_models

#------------------- Constants ------------------------------#

#------------------- Public Classes & Functions -------------#

def count(args):
    """
    K-mer frequency calculation
    """
    if (len(args.i) != 0):
        ## Check that the FASTA files exist
        if (not _validFastaFiles(args.i)):
            raise Exception("Invalid files.")

        print("Calculating K-mer frequencies")
        with SparkSession.builder.config(conf=_getSparkConf(args.t)).getOrCreate() as spark:
            calculate_kmer_frequencies.main(args.i, args.k, args.o)

def predict(args):
    """
    Predict the engineering status of DNA sequences
    """
    ## Check that the AVRO FREQ directories exist
    if (not _validAvroDirs(args.f)):
        raise Exception("Invalid directories.")

    ## Run a prediction using the provided model
    print("Predicting sequence class")
    predict_engineered_sequences.main(args.f, [args.m], args.o)

#------------------- Private Classes & Functions ------------#

def _getParser():
    def _addCountArgs(p, f):
        p.add_argument('-i', dest='i', help='FASTA/FASTQ files (.fa/.fq)',
            nargs='+', type=str, required=True)
        p.add_argument('-k', dest='k', help='k-mer length',
            type=int, required=True)
        p.add_argument('-o', dest='o', help='Directory containing K-mer frequencies (.avro)',
            type=str, required=True)
        p.add_argument('-t', dest='t', help='Number of partitions',
            nargs='?', type=int, default=8, required=False)
        p.set_defaults(func=f)

    def _addPredictArgs(p, f):
        p.add_argument('-f', dest='f', help='Directory containing K-mer frequencies (.avro)',
            nargs='+', type=str, required=True)
        p.add_argument('-m', dest='m', help='Predictive model (KerasClassifier)',
            type=str, required=True)
        p.add_argument('-o', dest='o', help='Output file (.tsv)',
            type=str, required=True)
        p.set_defaults(func=f)

    parser    = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    c = subparser.add_parser('count')
    p = subparser.add_parser('predict')
    _addCountArgs(c, count)
    _addPredictArgs(p, predict)
    return parser

def _getSparkConf(t):
    pfx = os.path.dirname(os.path.realpath(__file__)) + '/resources'
    p = pfx + "/spark-avro_2.12-3.3.1.jar"
    q = pfx + "/src.zip"
    conf = (
        SparkConf()
        .set("spark.jars", p)
        .set("spark.submit.pyFiles", q)
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
    parser = _getParser()
    args   = parser.parse_args()
    args.func(args)

#------------------------------------------------------------------------------
