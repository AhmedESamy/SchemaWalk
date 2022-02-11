import warnings  
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

import argparse
import gensim
import glob
import numpy as np
import pathlib
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle

import datasets

# classifier used in the original DeepWalk paper,
# taken from https://github.com/phanein/deepwalk/blob/master/example_graphs/scoring.py
# with only minimal changes (removed an assertion, changed numpy to np)
class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def evaluate(Xtrain, Ytrain, Xtest, Ytest):
    classifier = TopKRanker(LogisticRegression())
    classifier.fit(Xtrain, Ytrain)
    Ypred = classifier.predict(Xtest, Ytest.sum(axis = 1))
    Ypred = MultiLabelBinarizer(classes = classifier.classes_).fit_transform(Ypred)
    return (
        f1_score(Ytest, Ypred, average = 'micro'),
        f1_score(Ytest, Ypred, average = 'macro')
    )


def do_evaluate(filename, dataset, seeds):
    nodes, labels = datasets.load_labels(dataset)
    embs = gensim.models.KeyedVectors.load(filename)[nodes]

    percs = np.arange(1, 9) * .1
    cuts = (percs * len(nodes)).astype(int)
    test_cut = cuts[-1]

    micro = np.empty((len(cuts), len(seeds)))
    macro = np.empty((len(cuts), len(seeds)))
    for s, seed in enumerate(seeds):
        X, Y = shuffle(embs, labels, random_state = seed)
        for c, train_cut in enumerate(cuts):
            micro[c,s], macro[c,s] = evaluate(X[:train_cut], Y[:train_cut], X[test_cut:], Y[test_cut:])

    return percs, micro.mean(axis = 1), macro.mean(axis = 1)


def eprint(*args):
    print(*args, file = sys.stderr, flush = True)

def extract_dataset(filepath):
    dataset = filepath.stem.split('_')[0]
    if dataset in datasets.dataset_params:
        return dataset
    else:
        eprint(f'Could not automatically infer dataset for embedding file {filename} . {dataset} is not a valid dataset name. Please manually specify dataset with option --dataset')
        return None

def extract_seed(filepath, dataset):
    seed = filepath.stem.split('_')[-1]
    try:
        seed = int(seed)
        assert seed >= 1
    except:
        eprint(f'Could not automatically infer seed set for embedding file {filename} . {seed} is not a valid seed set. Please manually specify seed set with option --seed')
        return None
    if seed > len(datasets.dataset_params[dataset]['seeds']):
        eprint(f'Could not automatically infer seed set for embedding file {filename} . {dataset} has less than {seed} seed sets. Please manually specify seed set with option --seed')
        return None
    return seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluate node embeddings on node classification task using k-fold splits with pre-defined seeds\n\n'
                                                   'WARNING: for the dataset and seed to be automatically inferred, and therefore to be able to process multiple embedding files '
                                                   'in one call, it is necessary to use the default naming scheme from skipgram.py, i.e. <dataset>_<walk_type>_<run>.kv',
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('embeddings_pattern', type = str, nargs = '?', default = '*.kv', help = 'Pattern used to find all embedding files that should be evaluated (default *.kv)')
    parser.add_argument('--dataset', type = str, help = 'Manually specify which dataset the embeddings belong to; by default, it is inferred from the filename')
    parser.add_argument('--seed', type = int, help = 'Which set of 10 seeds to use for these embeddings; by default, it is inferred from the filename')
    parser.add_argument('-k', type = int, default = 10, help = 'Number of dataset folds to use; each seed set must contain at least as many seeds as the number of folds requested (default 10)')
    args = parser.parse_args()

    if args.dataset is not None and args.dataset not in datasets.dataset_params:
        eprint(f'{args.dataset} is not a valid dataset (dataset was provided manually with option --dataset)')
        sys.exit(1)

    for filename in glob.iglob(args.embeddings_pattern):
        path = pathlib.Path(filename)

        dataset = args.dataset or extract_dataset(path)
        if dataset is None:
            continue

        if args.seed is not None and args.seed > len(datasets.dataset_params[dataset]['seeds']):
            eprint(f'{filename} : {dataset} has less than the {args.seed} seed sets (seed set was provided manually with option --seed)')
            continue
        seedset = args.seed or extract_seed(path, dataset)
        if seedset is None:
            continue

        seeds = datasets.dataset_params[dataset]['seeds'][seedset - 1]
        if len(seeds) < args.k:
            eprint(f'Could not evaluate embedding file {filename} . Seed set {seedset} for dataset {dataset} has less than {args.k} seeds.')
            continue

        perc, micro, macro = do_evaluate(filename, dataset, seeds[:args.k])
        for p, mi, ma in zip(perc, micro, macro):
            print(f'{filename}\t{round(p,3)}\t{mi}\t{ma}')
