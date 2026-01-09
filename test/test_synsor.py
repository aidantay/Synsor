#!/bin/python

#------------------- Description & Notes --------------------#

#------------------- Dependencies ---------------------------#

# Standard library imports
from pathlib import Path

# External imports
import pytest

# Internal imports
from Synsor import make_parser
from Synsor import is_valid_parser
from Synsor import run

#------------------- Constants ------------------------------#

TEST_DATA_FASTA        = 'test/data/seqs.fna'
TEST_DATA_2MER_AVRO    = 'test/data/freqs_avro_2mer'
TEST_DATA_2MER_PARQUET = 'test/data/freqs_parquet_2mer'

#------------------- Classes & Functions --------------------#

def test_valid_i():
    parser = make_parser()
    args   = parser.parse_args([
        "-m", "model",
        "-o", "output",
        "-i", TEST_DATA_FASTA,
        "-k", "5",
    ])
    assert args.k == 5
    assert len(args.fnas) == 1
    assert args.fnas[0] == Path(TEST_DATA_FASTA)

def test_valid_I_avro():
    parser = make_parser()
    args   = parser.parse_args([
        "-m", "model",
        "-o", "output",
        "-I", TEST_DATA_2MER_AVRO,
    ])
    assert len(args.freqs) == 1
    assert args.freqs[0] == Path(TEST_DATA_2MER_AVRO)

def test_valid_I_parquet():
    parser = make_parser()
    args   = parser.parse_args([
        "-m", "model",
        "-o", "output",
        "-I", TEST_DATA_2MER_PARQUET,
        "-k", "5",
    ])
    assert len(args.freqs) == 1
    assert args.freqs[0] == Path(TEST_DATA_2MER_PARQUET)

def test_invalid_k():
    parser = make_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args([
            "-m", "model",
            "-o", "output",
            "-i", TEST_DATA_FASTA,
            "-k", "0"
        ])

def test_invalid_f():
    parser = make_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args([
            "-m", "model",
            "-o", "output",
            "-I", TEST_DATA_2MER_AVRO,
            "-f", "csv",
        ])

def test_invalid_i_I():
    parser = make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-m", "model",
            "-o", "output",
            "-i", TEST_DATA_FASTA,
            "-I", TEST_DATA_2MER_AVRO,
        ])

def test_invalid_I_k():
    parser = make_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args([
            "-m", "model",
            "-o", "output",
            "-I", TEST_DATA_2MER_AVRO,
            "-k", "4"
        ])
        is_valid_parser(parser, args)

#------------------- Main -----------------------------------#
