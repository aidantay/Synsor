# Synsor #

Synsor is a tool for identifying engineered DNA sequences. To accomplish this, Synsor leverages k-mer signature differences between natural and engineered DNA sequences and 
uses an artificial neural network to predict which sequences are likely to have been engineered. The model was trained on the 7-mer frequencies of natural plasmid and synthetic vector sequences from NCBI.

## Installation ##

Create the conda environment: `conda create env -f environment.yml`.

## Usage ##

### Calculate k-mer frequencies ###
```
python synsor.py \
    count \
    -i test.fa \
    -k 7 \
    -o freq_dir
```

### Predict engineered status of DNA sequences ###
```
python synsor.py \
    predict \
    -f freq_dir \
    -m model \
    -o output
```

## Contribution guidelines ##

The source codes are licensed under GPL less public licence. Users can contribute by making comments on the issues tracker, the wiki or direct contact via e-mail (see below).

## Publication ##

For more information, please refer to the following article:

Aidan P. Tay, Kieran Didi, Anuradha Wickramarachchi, Denis C. Bauer, Laurence O. W. Wilson, Maciej Maselko.<br>
**Synsor: a tool for alignment-free detection of engineered DNA sequences.**<br>
*Frontiers in Bioengineering and Biotechnology*, 2024, 12:1375626.<br>
[PMID:39070163](https://pubmed.ncbi.nlm.nih.gov/39070163)

## Contact ##

Aidan Tay: a.tay@unswalumni.com
