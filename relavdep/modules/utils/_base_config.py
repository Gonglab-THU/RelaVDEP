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