import numpy as np
from Bio import SeqIO

int2A = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 
         7: 'G', 8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 
         14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V'}
A2int = {value: key for key, value in int2A.items()}
int2AA = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'GLU', 4: 'ASP', 5: 'PHE', 6: 'ILE', 
          7: 'HIS', 8: 'LYS', 9: 'MET', 10: 'LEU', 11: 'ASN', 12: 'GLN', 13: 'PRO', 
          14: 'SER', 15: 'ARG', 16: 'THR', 17: 'TRP', 18: 'VAL', 19: 'TYR'}
AA2int = {value: key for key, value in int2AA.items()}
aa_list = list("ARNDCQEGHILKMFPSTWYV")

def read_fasta(fasta):
    FastaIterator = SeqIO.parse(fasta, "fasta")
    names, sequences = [], []
    for item in FastaIterator:
        names.append(item.id)
        sequences.append(''.join([s for s in item.seq]))
    assert len(sequences) == 1, "Input fasta file must contain only one sequence!"
    for s in sequences[0]:
        assert s in list(A2int.keys()), "Non-standard residue is not allowed in sequence!"
    return names[0], sequences[0]

def mutation(curr_seq, action):
    mutant = list(curr_seq)
    mut_pos, mut_res = (action - 1) // 20, (action - 1) % 20
    mutant[mut_pos] = int2A[mut_res]
    mutant = ''.join(mutant)
    return mutant

def seq2onehot(seq):
    onehot = np.zeros((len(seq), 20))
    for i in range(len(seq)):
        onehot[i, A2int[seq[i]]] = 1
    return onehot

class BaseConfig:
    def __init__(self):
        # MCTS
        self.discount = 0.99
        self.pb_c_base = 19652
        self.dirichlet_alpha = 0.25
        self.exploration_fraction = 0.25

        # reward model
        self.structure = None
        self.graph = None
        self.landscape = None

        # network
        self.hidden_dim = 128
        self.dropout = 0.1
        self.node_out_dim = 32
        self.node_in_dim = (128, 2)
        self.node_h_dim = (128, 16)
        self.edge_in_dim = (32, 1)
        self.edge_h_dim = (32, 1)
        self.n_layers = 3
        self.support_size = 10

        # trainer
        self.training_steps = 20000
        self.learning_rate = 5e-4
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 1000
        self.checkpoint_interval = 200
        self.weight_decay = 1e-4
        self.value_loss_weight = 1.0
        self.test_delay = 5
        self.play_delay = 0.1
        self.reanalyse_delay = 10
        
        # player
        self.temp_threshold = None

        # replay buffer
        self.buffer_size = 4000
        self.prob_alpha = 0.5

        # devices
        self.train_on_gpu = True
        self.test_on_gpu = True
        self.play_on_gpu = True
        self.predict_on_gpu = True
        self.reanalyse_on_gpu = True
