import argparse
import itertools
import multiprocessing

import datasets
import walkers

walk_types = {
    'Bidirectional': walkers.bidirectional,
    'DeepWalk': walkers.deepwalk,
    'JUST': walkers.just,
    'Metapath': walkers.metapath,
    'SchemaWalk': walkers.schemawalk,
    'SchemaWalkHO': walkers.schemawalk_ho,
}

def parallel_walks(outfile, workers, B, walk_func, walk_iter, static_args):
    with open(outfile, 'w') as walks_file, \
         multiprocessing.Pool(workers, _init_workers, (walk_func, static_args)) as pool:
        for walks_batch in pool.imap_unordered(_run_worker, batch_iterator(walk_iter, B)):
            walks_file.write(walks_batch)
        pool.close()
        pool.join()

def batch_iterator(iterator, batch_size):
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if len(batch) == 0:
            return
        yield batch

_worker_func = None
_worker_args = None
def _init_workers(func, args):
    global _worker_func, _worker_args
    _worker_func = func
    _worker_args = args

def _run_worker(batch_args):
    walks = ''
    for args in batch_args:
        walk = _worker_func(*args, *_worker_args)
        walks += ' '.join(walk) + '\n'
    return walks

def do_walks(walk_type, dataset, output_file, N = 80, L = 40, alpha = 0.4, beta = 0.1, iters = 12, mem = 2, workers = 4, B = 200):
    graph = datasets.load_graph(dataset)
    params = datasets.dataset_params[dataset]
    walk_func, walk_iter, static_args = walk_types[walk_type](graph, params, N, L, alpha, beta, iters, mem)
    parallel_walks(output_file, workers, B, walk_func, walk_iter, static_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate random walks on a graph')
    parser.add_argument('walk_type', type = str, help = 'Technique to use to generate random walks (must be listed in walk.py)')
    parser.add_argument('dataset', type = str, help = 'Name of the dataset to use (must be listed in datasets.py)')
    parser.add_argument('output_file', type = str, nargs = '?', default = None, help = 'Path to the output file where the random walks will be saved (defaults to <dataset>_walks.txt)')
    parser.add_argument('-N', type = int, default = 80, help = 'Number of random walks per node (default 80)')
    parser.add_argument('-L', type = int, default = 40, help = 'Length of each random walk (default 40)')
    parser.add_argument('--alpha', type = float, default = 0.4, help = 'Alpha hyperparameter (stay probability for JUST, decay rate for SchemaWalk and SchemaWalkHO) (default 0.4)')
    parser.add_argument('--beta', type = float, default = 0.1, help = 'Beta hyperparameter (Katz distance weight for SchemaWalkHO) (default 0.1)')
    parser.add_argument('--iters', type = int, default = 12, help = 'Number of iterations when computing the Katz similarity matrix (for SchemaWalkHO) (default 12)')
    parser.add_argument('--mem', type = int, default = 2, help = 'Memory length (for JUST) (default 2)')
    parser.add_argument('--workers', type = int, default = 4, help = 'Number of parallel workers for computing random walks (default 4)')
    parser.add_argument('-B', type = int, default = 200, help = 'Batch size for splitting walks across parallel workers (default 200)')
    args = parser.parse_args()

    output_file = args.output_file or f'{args.dataset}_walks.txt'
    do_walks(args.walk_type, args.dataset, output_file, args.N, args.L, args.alpha, args.beta, args.iters, args.mem, args.workers, args.B)
    