import numpy as np
from utils import *

class Config(BaseConfig):
    def __init__(self, args):
        super(Config, self).__init__()
        self.task_name, self.sequence = read_fasta(args.fasta)
        self.length = len(self.sequence)
        self.action_space = list(range(1, self.length * 20 + 1))
        self.action_space_size = len(self.action_space)
        if args.restraint:
            self.illegal = np.load(args.restraint)['illegal'].tolist()
            self.legal = np.load(args.restraint)['legal'].tolist()
        else:
            self.illegal = []
            self.legal = []
        self.output_path = args.output
        self.seed = args.seed

        self.n_sim = args.n_sim
        self.train_delay = args.train_delay
        self.max_mutations = args.max_mut
        self.data_dir = args.data_dir
        self.rm_params = args.rm_params
        self.rm_type = args.rm_type
        self.n_players = args.n_players
        self.batch_size = args.batch_size
        self.n_gpus = args.n_gpus
        self.no_buffer = args.no_buffer
        self.n_layer = args.n_layer

        if not args.unroll_steps:
            self.num_unroll_steps = args.max_mut // 2
        if not args.td_steps:
            self.td_steps = args.max_mut
    
    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25
