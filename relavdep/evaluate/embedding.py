import os
import timeit
import argparse
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import sys
sys.path.insert(0, './Dense-Homolog-Retrieval')
from mydpr.model.biencoder import MyEncoder
from mydpr.dataset.cath35 import PdDataModule

parser = argparse.ArgumentParser(description='Generate DHR embeddings for all mutants')
parser.add_argument('--mutants', type=str, required=True, help='Mutation data')
parser.add_argument('--output', type=str, required=True, help='Output directory')
args = parser.parse_args()

assert os.path.exists(args.mutants), "Mutation data does not exist!"
os.makedirs(args.output, exist_ok=True)

####################### Part 1 #######################
raw_data = pd.read_csv(args.mutants)
all_mutants = list(raw_data['mutant'])
all_sequences = list(raw_data['sequence'])
all_fitness = list(raw_data['fitness'])

####################### Part 2 #######################
print(f"====== Generate DHR embeddings for {len(raw_data)} mutants ======")
s_time = timeit.default_timer()
ckpt_path = os.path.dirname(os.path.abspath(__file__))
batch_size = 100
model = MyEncoder(bert_path=[os.path.join(ckpt_path, 'Dense-Homolog-Retrieval/dhr_qencoder.pt'), 
                             os.path.join(ckpt_path, 'Dense-Homolog-Retrieval/dhr_cencoder.pt')])
trainer = pl.Trainer(devices=[0], accelerator="gpu", accumulate_grad_batches=4, precision=32, fast_dev_run=False)

seqs = [''.join([x.upper() for x in s if x.isalpha()]) for s in all_sequences]
df = pd.DataFrame({"id": np.arange(0, len(seqs)), "seqs": seqs})
df.to_csv(f'{args.output}/pre_embedding.tsv', sep='\t', header=None, index=False)
dm = PdDataModule(f'{args.output}/pre_embedding.tsv', batch_size, model.alphabet, trainer)
output = trainer.predict(model, datamodule=dm)
output = torch.cat(output, dim=0)
data = {"mutant": all_mutants, "sequence": all_sequences, "embedding": output, "fitness": all_fitness}
torch.save(data, f'{args.output}/embeddings.pt')

e_time = timeit.default_timer()
print(">>> Task finished. Execution time: {:.2f}s <<<".format(e_time - s_time))
