# Synsor

Synsor is a tool for identifying engineered DNA sequences. To accomplish this, Synsor leverages k-mer signature differences between natural and engineered DNA sequences and uses an artificial neural network to predict which sequences are likely to have been engineered. The model was trained on the 7-mer frequencies of natural plasmid and synthetic vector sequences from NCBI.

## Installation

```
## Clone repo
git clone git@bitbucket.org:aidantay/Synsor.git
cd Synsor

## Create and activate conda environment:
conda env create -f environment.yml
conda activate synsor
```

## Usage

### via Python

```
python Synsor.py \
    -i test/data/seqs.fna \
    -k 7 \
    -m model \
    -o Synsor.results
```

### via Spark

```
spark-submit \
    --driver-memory '8g' \
    --jars resources/spark-avro_2.12-3.4.1.jar \
    --conf spark.default.parallelism=8 \
    Synsor.py \
        -i test/data/seqs.fna \
        -k 7 \
        -m model \
        -o Synsor.results \
        -f avro
```

## All options

```
usage: Synsor.py [-h] (-i FNAS [FNAS ...] | -I FREQS [FREQS ...]) [-k K] -m MODEL -o OUTDIR [-f {parquet,avro}]

Detect engineered DNA sequences in high-throughput sequencing data sets

options:
  -h, --help            show this help message and exit

INPUT PARAMETERS:
  -i FNAS [FNAS ...], --fnas FNAS [FNAS ...]
                        FASTA/FASTQ files (.fa/.fq)
  -I FREQS [FREQS ...], --freqs FREQS [FREQS ...]
                        Directories containing k-mer frequencies
  -k K                  k-mer length (cannot be used with -I)
  -m MODEL, --model MODEL
                        Predictive model (KerasClassifier)

OUTPUT PARAMETERS:
  -o OUTDIR, --outdir OUTDIR
                        Output directory

ADDITIONAL PARAMETERS:
  -f {parquet,avro}, --output-fmt {parquet,avro}
                        Output format for k-mer frequencies [parquet]

```

## Contribution guidelines

The source codes are licensed under GPL less public licence. Users can contribute by making comments on the issues tracker, the wiki or direct contact via e-mail (see below).

## Publication

For more information, please refer to the following article:

Aidan P. Tay, Kieran Didi, Anuradha Wickramarachchi, Denis C. Bauer, Laurence O. W. Wilson, Maciej Maselko.<br>
**Synsor: a tool for alignment-free detection of engineered DNA sequences.**<br>
*Frontiers in Bioengineering and Biotechnology*, 2024, 12:1375626.<br>
[PMID:39070163](https://pubmed.ncbi.nlm.nih.gov/39070163)

## Contact

* Aidan Tay: a.tay@unswalumni.com
