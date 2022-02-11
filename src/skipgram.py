import warnings  
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

import argparse
import gensim
import os

from walk import do_walks

def do_skipgram(walks_file, embeddings_file, D, k, ns, e, workers):
    skipgram = gensim.models.Word2Vec(corpus_file = walks_file, vector_size = D, window = k, negative = ns, epochs = e, sg = 1, workers = workers)
    skipgram.wv.save(embeddings_file)

if __name__ == '__main__':
    skipgram_parser = argparse.ArgumentParser(add_help = False)
    skipgram_parser.add_argument('--workers', type = int, default = 4, help = 'Number of parallel workers for computing random walks (default 4)')
    skipgram_args = skipgram_parser.add_argument_group('SkipGram arguments')
    skipgram_args.add_argument('-D', type = int, default = 128, help = 'Embedding dimensionality')
    skipgram_args.add_argument('-k', type = int, default = 5, help = 'Context size (default 5)')
    skipgram_args.add_argument('--ns', type = int, default = 5, help = 'Number of negative samples (default 5)')
    skipgram_args.add_argument('-e', type = int, default = 5, help = 'Number of epochs (default 5)')

    parser = argparse.ArgumentParser(description = 'Generate random walk-based node embeddings using SkipGram')
    subparsers = parser.add_subparsers(title = 'Subcommands', dest = 'command', description = 'Use `<command_name> --help` to see the options for a specific subcommand')

    from_walks = subparsers.add_parser('from-walks', parents = [skipgram_parser], help = 'Build embeddings from pre-computed random walks')
    from_walks.add_argument('walks_file', type = str, help = 'Path to a file containing random walks (generated using walk.py)')
    from_walks.add_argument('embeddings_file', type = str, nargs = '?', default = 'embeddings.kv', help = 'Path to a file where the embeddings will be saved (default embeddings.kv)')

    from_graph = subparsers.add_parser('from-graph', parents = [skipgram_parser], help = 'Build one or more embedding files from random walks generated on-the-fly')
    from_graph.add_argument('walk_type', type = str, help = 'Technique to use to generate random walks (must be listed in walk.py)')
    from_graph.add_argument('dataset', type = str, help = 'Name of the dataset to use (must be listed in datasets.py)')
    from_graph.add_argument('embeddings_file_prefix', type = str, nargs = '?', help = 'Prefix for output files where embeddings will be saved (default: <dataset>_<walk_type>_)')
    from_graph.add_argument('-C', type = int, default = 1, help = 'How many independent runs to perform, i.e. how many embedding files to output; each output file will use the same prefix, with a numerical suffix starting from one')
    from_graph.add_argument('--force', action = 'store_true', help = 'Force overwriting existing files when performing more than one run')

    walk_args = from_graph.add_argument_group('Walk arguments')
    walk_args.add_argument('-N', type = int, default = 80, help = 'Number of random walks per node (default 80)')
    walk_args.add_argument('-L', type = int, default = 40, help = 'Length of each random walk (default 40)')
    walk_args.add_argument('--alpha', type = float, default = 0.4, help = 'Alpha hyperparameter (stay probability for JUST, decay rate for SchemaWalk and SchemaWalkHO) (default 0.4)')
    walk_args.add_argument('--beta', type = float, default = 0.1, help = 'Beta hyperparameter (Katz distance weight for SchemaWalkHO) (default 0.1)')
    walk_args.add_argument('--iters', type = int, default = 12, help = 'Number of iterations when computing the Katz similarity matrix (for SchemaWalkHO) (default 12)')
    walk_args.add_argument('--mem', type = int, default = 2, help = 'Memory length (for JUST) (default 2)')
    walk_args.add_argument('-B', type = int, default = 200, help = 'Batch size for splitting walks across parallel workers (default 200)')
    walk_args.add_argument('--tmp', type = str, default = 'tmp-walks', help = 'Path of temporary file where the generated random walks will be stored (automatically deleted at program exit)')

    args = parser.parse_args()

    if args.command == 'from-walks':
        do_skipgram(args.walks_file, args.embeddings_file, args.D, args.k, args.ns, args.e, args.workers)
    else:
        prefix = args.embeddings_file_prefix or f'{args.dataset}_{args.walk_type}_'
        for i in range(1, args.C + 1):
            output_file = f'{prefix}{i}.kv'
            if args.force or not os.path.exists(output_file):
                print(f'Run {i} of {args.C}')
                do_walks(args.walk_type, args.dataset, args.tmp, args.N, args.L, args.alpha, args.beta, args.iters, args.mem, args.workers, args.B)
                do_skipgram(args.tmp, output_file, args.D, args.k, args.ns, args.e, args.workers)
            else:
                print(f'{output_file} already exists. Skipping this run.')
        os.remove(args.tmp)
