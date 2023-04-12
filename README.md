# README #

Synsor is a tool for identifying engineered DNA sequences. To accomplish this, Synsor leverages k-mer signature differences between natural and engineered DNA sequences and 
uses an artificial neural network to predict which sequences are likely to have been engineered. The model was trained on the 7-mer frequencies of natural plasmid and synthetic vector sequences from NCBI.

## Requirements ##
* python >= 3.7.0
* pyspark >= 3.0.0
* typer >= 0.7.0
* fastavro >= 1.7.3
* tensorflow >= 2.10.0
* scikit-learn >= 1.2.0

Alternatively, create the conda environment: `conda create env -f environment.yml`.

## Usage ##

### Without pre-computed K-mer frequencies ###
```
python synsor.py \
    --i test.fa \
    --f freqs \
    --m model \
    --o output
```

### With pre-computed K-mer frequencies ###
```
python synsor.py \
    --f freqs \
    --m model \
    --o output
```

## Contribution guidelines ##

The source codes are licensed under GPL less public licence. Users can contribute by making comments on the issues tracker, the wiki or direct contact via e-mail (see below).

## Contact ##

Aidan Tay: a.tay@unswalumni.com
