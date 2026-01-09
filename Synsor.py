#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
import argparse
import logging
from pathlib import Path

# External imports
from pyspark.sql import SparkSession

# Internal imports
from synsor import io
from synsor import model
from synsor import schema
from synsor import seq2kmer

#------------------- Constants ------------------------------#

logger = logging.getLogger(__name__)

#------------------- Classes & Functions --------------------#

def run(modelFile, oDir, fnas=None, freqs=None, kmerLength=None, oFormat='parquet'):
    ss = SparkSession.getActiveSession()
    sc = ss.sparkContext

    ''' READ DATA - CALCULATE KMER COUNTS IF APPLICABLE '''
    if (freqs):
        logger.info("Loading frequencies")
        kmerDf = io.kmer.read(*freqs)

    else:
        logger.info("Reading sequences")
        seqRecRdd = sc.parallelize(fnas, len(fnas))
        seqRecRdd = seqRecRdd.flatMap(io.fastx.read)
        seqRecRdd = seqRecRdd.repartition(sc.defaultParallelism)
        kmerDf    = seq2kmer.getCounts(seqRecRdd, kmerLength, True, False)

        ## Write the table to disk
        logger.info("Writing frequencies")
        d = oDir / 'freqs'
        io.kmer.write(d, kmerDf, oFormat)

    ''' PROCESS KMER COUNTS '''
    logger.info("Preparing frequencies")
    kmerDf = model.prepareData(kmerDf)

    ''' MAKE PREDICTION USING PROVIDED MODEL '''
    logger.info("Predicting engineer status")
    iDim   = [kmerDf.select(schema.KMER_COL_NAME).collect()[0][0].size]
    predDf = model.getPredictions(modelFile, kmerDf, iDim)

    ''' CLEAN UP AND GENERATE OUTPUTS '''
    logger.info("Writing results")
    d = oDir / 'predictions'
    predDf = predDf.coalesce(sc.defaultParallelism)
    predDf.write.csv(str(d), mode='overwrite', sep='\t', header=True, compression='gzip')
    logger.info("Done!")
    logger.info("Thank you for using Synsor!")

''' Parser '''

def make_parser():
    def k_gt_1(value):
        value = int(value)
        if value < 1:
            raise argparse.ArgumentTypeError("-k must be >= 1")
        return value

    parser = argparse.ArgumentParser(description='Detect engineered DNA sequences in high-throughput sequencing data sets')

    i = parser.add_argument_group('INPUT PARAMETERS')
    g = i.add_mutually_exclusive_group(required=True)
    g.add_argument('-i', '--fnas'      , type=Path  , default=None     , nargs='+' , help='FASTA/FASTQ files (.fa/.fq)')
    g.add_argument('-I', '--freqs'     , type=Path  , default=None     , nargs='+' , help='Directories containing k-mer frequencies')
    i.add_argument('-k'                , type=k_gt_1, default=None     , nargs=None, help='k-mer length (cannot be used with -I)')
    i.add_argument('-m', '--model'     , type=Path  , default=None     , nargs=None, help='Predictive model (KerasClassifier)', required=True)

    o = parser.add_argument_group('OUTPUT PARAMETERS')
    o.add_argument('-o', '--outdir'    , type=Path  , default=None     , nargs=None, help='Output directory'                  , required=True)

    a = parser.add_argument_group('ADDITIONAL PARAMETERS')
    a.add_argument('-f', '--output-fmt', type=str   , default='parquet', nargs=None, help='Output format for k-mer frequencies [parquet]', choices=["parquet", "avro"])
    return parser

def is_valid_parser(parser, args):
    if (args.freqs and args.k):
        parser.error("-k/ not allowed with -I/--freqs")

''' Logging '''

def setup_logging(log_dir):
    handlers = [logging.StreamHandler()]

    if (log_dir is not None):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "Synsor.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s [%(levelname)s] Synsor: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

#------------------- Main -----------------------------------#

if (__name__ == "__main__"):
    parser = make_parser()
    args   = parser.parse_args()
    is_valid_parser(parser, args)

    setup_logging(args.outdir)
    with SparkSession.builder.getOrCreate() as ss:
        ss.sparkContext.setLogLevel("ERROR")
        with ss.sparkContext as sc:
            logger.info("Starting program")
            logger.info("Parsed arguments: %s", args)
            run(args.model, args.outdir, args.fnas, args.freqs, args.k, args.output_fmt)

#------------------------------------------------------------------------------
