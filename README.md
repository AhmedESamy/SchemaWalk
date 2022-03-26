# SchemaWalk

This repository contains the code for _SchemaWalk: Schema Aware Random Walks for Heterogeneous Graph Embedding_, a paper **recently accepted in WWW, Companion Volume**.

## Contents

This repository includes:
* Edge lists and node labels of four heterogeneous graphs:
  - ACM
  - DBLP
  - Foursquare
  - Movies
* Code to generate random walks on those datasets using any of the following techniques:
  - DeepWalk
  - Bidirectional walks
  - Metapath2vec
  - JUST
  - SchemaWalk
  - SchemaWalkHO
* Code to train SkipGram embeddings based on those random walks
* Code to evaluate the micro and macro F1 scores of those embeddings on the node classification task defined by the labels in the datasets

## Installation

To ensure reproducibility, the code was tested inside a `conda` environment. The list of packages installed is available in `environment.yml` and the environment can be easily rebuilt either manually or with the following command:
```
conda env create -f environment.yml
```

## Usage

Each script provides extensive command line options that can be listed by providing the `--help` flag either to the script itself or to one of its specific sub-command.

* To generate a file containing random walks for a given dataset (e.g. DBLP) with a given technique (e.g. SchemaWalk) use:
  ```
  src/walk.py SchemaWalk DBLP
  ```
  Several flags are available to control the hyperparameters of the random walk

* To generate SkipGram embeddings starting from a pre-computed random walks file, use:
  ```
  src/skipgram.py from-walks walks_file.txt
  ```
  Several flags are available to control the hyperparameters of SkipGram

* To generate several SkipGram embeddings, each computed on a separate set of random walks that is generated on-the-fly, with a given technique (e.g. DeepWalk) on a given dataset (e.g. Movie), use:
  ```
  src/skipgram.py from-graph DeepWalk Movie
  ```
  This sub-command accepts options to control the hyperparameters of both the random walk and SkipGram

* To evaluate an embeddings file obtained on a given dataset (e.g. Foursquare), using one of the available seed sets (e.g. seed set 3) use:
  ```
  src/evaluate.py embeddings.kv --dataset Foursquare --seed 3 > evaluation_results.tsv
  ```
  The flags allow specifying the dataset, the seed set (all seed sets are listed in `src/datasets.py`) and how many data splits to use.

* To evaluate all embedding files in the current directory, as obtained by running the `from-graph` sub-command explained before, simply use:
  ```
  src/evaluate.py > evaluation_results.tsv
  ```

## How the Evaluation is Performed

Each embeddings file passed to `src/evaluate.py` is evaluated multiple times, on different data splits. Each data split is obtained by shuffling the data with one of the seeds in the chosen seed set (all seed sets are listed in `src/datasets.py`). The micro and macro F1 scores reported are the average across all data splits.

For each data split, the last 20% of the data is used for evaluating 8 classifiers, trained on the first 10%, 20%, ..., 80% of the data respectively. The micro and macro F1 scores are reported separately for each training data percentage.
