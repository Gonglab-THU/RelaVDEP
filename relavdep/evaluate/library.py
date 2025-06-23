import os
import ray
import random
import argparse
import timeit
import torch
import logomaker
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import binned_statistic_2d
from scipy.stats import entropy
from utils import *

parser = argparse.ArgumentParser(description='Construct mutant library')
parser.add_argument('--fasta', type=str, required=True, help='Protein sequence')
parser.add_argument('--embedding', type=str, required=True,  help='DHR embeddings')
parser.add_argument('--output', type=str, required=True, help='Output directory')

parser.add_argument('--cutoff', type=float, default=0, help='Fitness cutoff (default: %(default)s)')
parser.add_argument('--size', type=int, default=10, help='Mutant library size (default: %(default)s)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: %(default)s)')
parser.add_argument('--n_cpu', type=int, default=10, help='Number of CPUs used in parallel (default: %(default)s)')
args = parser.parse_args()

assert os.path.exists(args.fasta), "Protein sequence does not exist!"
assert os.path.exists(args.embedding), "DHR embeddings do not exist, please run 'embedding.py'!"
os.makedirs(args.output, exist_ok=True)
target, wt_seq = read_fasta(args.fasta)
raw_data = torch.load(args.embedding)
assert args.cutoff < raw_data['fitness'][args.size], "Inappropriate cutoff!"
cutoff_index = np.where(np.array(raw_data['fitness']) > args.cutoff)[0][-1] + 1
sele_mutants = raw_data['mutant'][:cutoff_index]
sele_sequences = raw_data['sequence'][:cutoff_index]
sele_embeddings = raw_data['embedding'][:cutoff_index]
sele_fitness = raw_data['fitness'][:cutoff_index]
data_df = pd.DataFrame({"mutant": sele_mutants, "sequence": sele_sequences, "fitness": sele_fitness})

print(f"========== Construct the library from {len(data_df)} mutants ==========")
assert args.size >= 10, "The size of library cannot be less than 10!"
assert args.size <= len(data_df), "The size of library must be less than the number of the selected mutants!"
ray.init(num_cpus=args.n_cpu)

print("Step 1: Performing t-SNE on DHR embeddings...")
s_time = timeit.default_timer()
tsne = TSNE(n_components=2, random_state=args.seed)
tsne_result = tsne.fit_transform(sele_embeddings)

print("Step 2: Selecting the best cluster number...")
best_k, best_score = 0, -1
for k in range(4, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=args.seed).fit(sele_embeddings)
    score = silhouette_score(sele_embeddings, kmeans.labels_)
    if score > best_score:
        best_score = score
        best_k = k

def init_library(cluster_labels):
    data = pd.DataFrame({'cluster': cluster_labels, 'sequence': sele_sequences, 'fitness': sele_fitness})
    selected_sequences, selected_fitness = [], []
    top_in_each_cluster = data.loc[data.groupby('cluster')['fitness'].idxmax()]
    selected_sequences.extend(top_in_each_cluster['sequence'].tolist())
    selected_fitness.extend(top_in_each_cluster['fitness'].tolist())
    remaining_size = args.size - len(selected_sequences)
    if remaining_size > 0:
        cluster_fitness_mean = data.groupby('cluster')['fitness'].mean()
        cluster_weights = cluster_fitness_mean / cluster_fitness_mean.sum()
        additional_allocation = (cluster_weights * remaining_size).astype(int)
        remaining_to_allocate = remaining_size - additional_allocation.sum()
        if remaining_to_allocate > 0:
            sorted_clusters = cluster_weights.sort_values(ascending=False).index
            for cluster in sorted_clusters:
                if remaining_to_allocate == 0:
                    break
                additional_allocation[cluster] += 1
                remaining_to_allocate -= 1
        
        for cluster, count in additional_allocation.items():
            if count > 0:
                cluster_data = data[data['cluster'] == cluster].sort_values('fitness', ascending=False)
                additional_sequences = cluster_data['sequence'].tolist()[1:count+1]
                additional_fitness = cluster_data['fitness'].tolist()[1:count+1]
                selected_sequences.extend(additional_sequences)
                selected_fitness.extend(additional_fitness)
    return selected_sequences[:args.size], selected_fitness[:args.size]

def get_mutation_frequencies(sequences):
    mutation_frequencies = []
    for i in range(len(wt_seq)):
        ref_aa = wt_seq[i]
        mutated_aa_list = [seq[i] for seq in sequences]
        mutated_aa_counts = Counter(mutated_aa_list)
        if any(aa != ref_aa for aa in mutated_aa_list):
            total_mutants = len(sequences)
            frequencies = {aa: count / total_mutants for aa, count in mutated_aa_counts.items()}
            mutation_frequencies.append(frequencies)
        else:
            mutation_frequencies.append({ref_aa: 1.0})
    return mutation_frequencies

def objective_function(sequences, fitness, lam):
    mutation_frequencies = get_mutation_frequencies(sequences)
    mutation_matrix = []
    mutation_pos = []
    for i in range(len(wt_seq)):
        freqs = mutation_frequencies[i]
        if len(list(freqs.keys())) > 1:
            row = [freqs.get(aa, 0) for aa in aa_list]
            mutation_matrix.append(row)
            mutation_pos.append(i+1)
    
    diversity = 0
    for res in range(len(mutation_matrix)):
        diversity += entropy(mutation_matrix[res], base=2)
    
    objective = np.mean(fitness) + lam * diversity
    return diversity, objective, mutation_matrix, mutation_pos

@ray.remote
def optimization(sequences, fitness, lam, seed, iterations=2000):
    random.seed(seed)
    current_sequences = sequences.copy()
    current_fitness = np.array(fitness)
    current_diversity, current_objective, _, _ = objective_function(current_sequences, current_fitness, lam)

    fitness_his, diversity_his = [], []
    for _ in range(iterations):
        idx = random.randint(0, len(sequences) - 1)

        new_sequences = current_sequences.copy()
        candidate = random.choice(sele_sequences)
        while candidate in current_sequences:
            candidate = random.choice(sele_sequences)
        new_sequences[idx] = candidate
        
        new_fitness = current_fitness.copy()
        new_fitness[idx] = sele_fitness[sele_sequences.index(candidate)]
        new_diversity, new_objective, _, _ = objective_function(new_sequences, new_fitness, lam)

        if new_objective > current_objective:
            current_sequences = new_sequences
            current_fitness = new_fitness
            current_objective = new_objective
            current_diversity = new_diversity
            fitness_his.append(np.mean(new_fitness))
            diversity_his.append(new_diversity)
    return current_sequences, current_fitness, current_diversity

def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

print("Step 3: K-means clustering...")
kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=args.seed)
clusters = kmeans.fit_predict(sele_embeddings)

print("Step 4: Selecting variants...")
lambda_list = np.arange(0.01, 1.01, 0.01)
starting_sequences, starting_fitness = init_library(clusters)
iterations = max(len(data_df), 2000)
futures = [optimization.remote(starting_sequences, starting_fitness, lam, args.seed, iterations=iterations) for lam in lambda_list]

sequences_history, fitness_history, diversity_history = [], [], []
for result in ray.get(futures):
    sequences_history.append(result[0])
    fitness_history.append(result[1])
    diversity_history.append(result[2])

mc_result = pd.DataFrame({"lambda": lambda_list, "fitness": np.mean(fitness_history, axis=1), "diversity": diversity_history})
mc_result['fitness-norm'] = min_max_normalize(mc_result['fitness'])
mc_result['diversity-norm'] = min_max_normalize(mc_result['diversity'])
mc_result["area"] = mc_result['fitness-norm'] * mc_result['diversity-norm']

best_index = np.argmax(mc_result["area"])
best_lam = lambda_list[best_index]
selected_sequences = sequences_history[best_index]
selected_indices = [data_df[data_df['sequence'] == seq].index[0] for seq in selected_sequences]

library = data_df.loc[selected_indices].copy().sort_values(by="fitness", ascending=False)
library.to_csv(os.path.join(args.output, 'library.csv'), index=False)

print("Step 5: Ploting figures...")

sns.set_style('ticks')
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans'],
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 24, 
    'ytick.labelsize': 24,
    'figure.figsize': (8, 6),
    'savefig.bbox': 'tight',
    'savefig.transparent': False})

# library
x = tsne_result[:, 0]
y = tsne_result[:, 1]
z = sele_fitness

stat, x_edges, y_edges, binnumber = binned_statistic_2d(x, y, z, statistic='mean', bins=100)
plt.imshow(np.flipud(stat.T), extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
           cmap='RdBu_r', aspect='auto', interpolation='nearest', alpha=0.8)

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=24)
cbar.set_label("Predicted fitness", fontsize=26, rotation=270, labelpad=25)

plt.scatter([tsne_result[idx, 0] for idx in selected_indices],
            [tsne_result[idx, 1] for idx in selected_indices], 
            c='#963B79', s=30, marker='^')

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(args.output, 'library.png'), dpi=300)

# frequency
library_sequences = list(library['sequence'])
library_fitness = list(library['fitness'])
library_diversity, _, library_matrix, mutation_pos = objective_function(library_sequences, library_fitness, best_lam)
mutation_df = pd.DataFrame(library_matrix, columns=aa_list)

fig_length = max(len(mutation_pos) // 3, 10)
logo = logomaker.Logo(mutation_df, color_scheme='NajafabadiEtAl2017', 
                      shade_below=0.5, fade_below=0.5, figsize=([fig_length, 3]))

plt.title("Mutant library (diversity={:.2f}, fitness={:.2f}, lambda={:.2f})".format(
    library_diversity, np.mean(library_fitness), best_lam))

plt.xlabel("Residue index")
plt.xticks(np.arange(len(mutation_pos)), mutation_pos, rotation=45)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(args.output, 'frequency.png'), dpi=300)

e_time = timeit.default_timer()
print(">>> Task finished! Execution time: {:.2f}s <<<".format(e_time - s_time))
