import os, sys
import argparse
import torch
import pandas as pd
import numpy as np
import subprocess
import ray
import random
import logomaker
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.stats import binned_statistic_2d
from scipy.stats import entropy

from relavdep.modules.utils._functions import *
from relavdep.modules.utils._models import *
from scripts.SPIRED_Fitness.models import Model
from scripts.mutant_library.functions import *

parser = argparse.ArgumentParser(description='Construct mutant library')
parser.add_argument('--fasta', type=str, required=True, help='Protein sequence')
parser.add_argument('--mutants', type=str, required=True, help='Mutants data')

parser.add_argument('--output', type=str, default='outputs', help='Output directory (default: %(default)s)')
parser.add_argument('--cutoff', type=float, default=0, help='Fitness cutoff (default: %(default)s)')
parser.add_argument('--size', type=int, default=10, help='Mutant library size (default: %(default)s)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: %(default)s)')
parser.add_argument('--n_cpu', type=int, default=10, help='Number of CPUs used in parallel (default: %(default)s)')
args = parser.parse_args()

env_name = "fastMSA"
script = "scripts/mutant_library/extract_embeddings.py"
embeddings_path = f"{args.output}/dhr_embeddings"

def check_env(env_name):
    print(f"Checking for Conda environment: {env_name}...")
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"], 
            check=True, 
            capture_output=True, 
            text=True
        )

        if env_name in result.stdout:
            print(f"Environment '{env_name}' exists.")
            return True
        else:
            print(f"ERROR: Conda environment '{env_name}' not found!")
            return False
    except subprocess.CalledProcessError:
        print("ERROR: Failed to execute 'conda info --envs'. Please check if Conda is working properly.")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: 'conda' command not found. Please ensure Conda is properly installed and configured in your system's PATH.")
        sys.exit(1)

def run_first_stage():
    start_time = print_stage_start(1, 3, "Extracting DHR Embeddings")
    os.makedirs(embeddings_path, exist_ok=True)

    print(f"Input Mutants File: {args.mutants}")
    print(f"Output Directory: {embeddings_path}")

    command = [
        "conda", 
        "run", 
        "-n", 
        env_name, 
        "python", 
        script, 
        "--mutants", 
        args.mutants, 
        "--output", 
        embeddings_path
    ]

    try:
        subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("Embeddings extraction successful.")
        print_stage_end(1, start_time)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error code: {e.returncode}")
        print("--- Stdout ---\n", e.stdout)
        print("--- Stderr ---\n", e.stderr)
        print_stage_end(1, start_time, "FAILED")
        sys.exit(1)

def run_second_stage():
    start_time = print_stage_start(2, 3, "Library Construction & Optimization")

    raw_data = torch.load(os.path.join(embeddings_path, "embeddings.pt"))

    if args.cutoff >= raw_data['fitness'][args.size]:
        print(f"!!! Inappropriate cutoff !!!")
        print_stage_end(2, start_time, "FAILED")
        sys.exit(1)
    if args.size < 10:
        print(f"!!! The size of library cannot be less than 10 !!!")
        print_stage_end(2, start_time, "FAILED")
        sys.exit(1)

    cutoff_index = np.where(np.array(raw_data['fitness']) > args.cutoff)[0][-1] + 1

    sele_mutants = raw_data['mutant'][:cutoff_index]
    sele_sequences = raw_data['sequence'][:cutoff_index]
    sele_embeddings = raw_data['embedding'][:cutoff_index]
    sele_fitness = raw_data['fitness'][:cutoff_index]

    data_df = pd.DataFrame({"mutant": sele_mutants, "sequence": sele_sequences, "fitness": sele_fitness})

    if args.size > len(data_df):
        print(f"!!! Library size must be less than the number of selected mutants !!!")
        print_stage_end(2, start_time, "FAILED")
        sys.exit(1)

    print(f"\n--- Selection Summary ---")
    print(f"Fitness Cutoff: {args.cutoff}")
    print(f"Total Mutants Selected: {len(data_df)} (Fitness > Cutoff)")
    print(f"Target Library Size: {args.size}")
    print(f"Sequence Length: {len(target_sequence)}")
    print(f"Reference Protein: {target_name}")
    print("-" * 30)
    
    print("[Step 1/5] Performing t-SNE on DHR embeddings...")
    step_start = time.time()
    
    tsne = TSNE(n_components=2, random_state=args.seed)
    tsne_result = tsne.fit_transform(sele_embeddings)
    print(f"  -> t-SNE completed. Duration: {format_time(time.time() - step_start)}")
    
    print("[Step 2/5] Selecting the best cluster number (K=4 to 10)...")
    best_k, best_score = 0, -1
    for k in range(4, 11):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=args.seed).fit(sele_embeddings)
        score = silhouette_score(sele_embeddings, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    print(f"  -> Best Cluster Number (K): {best_k}")
    print(f"  -> Duration: {format_time(time.time() - step_start)}")

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

        for i in range(len(target_sequence)):
            ref_aa = target_sequence[i]
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
        for i in range(len(target_sequence)):
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

    print("[Step 3/5] K-means clustering with K={best_k}...")
    step_start = time.time()
    kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=args.seed)
    clusters = kmeans.fit_predict(sele_embeddings)
    print(f"  -> Clustering completed. Duration: {format_time(time.time() - step_start)}")

    print("[Step 4/5] Multi-objective optimization (Fitness & Diversity)...")
    ray.init(log_to_driver=False, _temp_dir='/tmp/ray', num_cpus=args.n_cpu)
    print(f"  -> Ray initialized with {args.n_cpu} CPUs.")

    step_start = time.time()
    lambda_list = np.arange(0.01, 1.01, 0.01)
    starting_sequences, starting_fitness = init_library(clusters)
    iterations = max(len(data_df), 2000)
    
    futures = [optimization.remote(starting_sequences, starting_fitness, lam, args.seed, iterations=iterations) for lam in lambda_list]

    sequences_history, fitness_history, diversity_history = [], [], []

    for result in tqdm(ray.get(futures), desc="Optimization Progress"):
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

    print(f"\n  -> Optimization completed. Duration: {format_time(time.time() - step_start)}")

    ray.shutdown()
    print("  -> Ray shutdown.")

    print("[Step 5/5] Ploting figures (library.png & frequency.png)...")

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

    x = tsne_result[:, 0]
    y = tsne_result[:, 1]
    z = sele_fitness

    stat, x_edges, y_edges, binnumber = binned_statistic_2d(x, y, z, statistic='mean', bins=50)
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

    library_sequences = list(library['sequence'])
    library_fitness = list(library['fitness'])
    library_diversity, _, library_matrix, mutation_pos = objective_function(library_sequences, library_fitness, best_lam)
    mutation_df = pd.DataFrame(library_matrix, columns=aa_list)

    fig_length = max(len(mutation_pos) // 3, 12)
    logomaker.Logo(
        mutation_df, color_scheme='NajafabadiEtAl2017', 
        shade_below=0.5, fade_below=0.5, figsize=([fig_length, 3])
    )

    plt.title("Mutant library (diversity={:.2f}, fitness={:.2f})".format(library_diversity, np.mean(library_fitness)))
    plt.xlabel("Residue index")
    plt.xticks(np.arange(len(mutation_pos)), mutation_pos, rotation=45)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'frequency.png'), dpi=300)

    print(f"  -> Figures saved to {args.output}")
    print(f"  -> Final Library Mean Fitness: {np.mean(library_fitness):.4f}")
    print(f"  -> Final Library Diversity: {library_diversity:.4f}")
    print(f"  -> Duration: {format_time(time.time() - step_start)}")

    print_stage_end(2, start_time)

    return library

def run_third_stage(library):
    start_time = print_stage_start(3, 3, "Stability Prediction (ΔΔG & ΔTm)")

    sele_mutants = library["sequence"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> Using device: {device}")

    print("[Step 1/2] Loading models...")
    step_start = time.time()

    base_model = BaseModel(data_dir="models", device=device)

    stab_model = Model(node_dim = 32, num_layer = 3, n_head = 8, pair_dim = 64)
    best_model = torch.load("models/SPIRED-Stab.pth", map_location=torch.device('cpu')).copy()
    best_dict = {k.split('Stab.')[-1]: v for k, v in best_model.items() if k.startswith('Stab')}
    stab_model.load_state_dict(best_dict)
    stab_model.eval().to(device)

    print(f"  -> Models loaded successfully. Duration: {format_time(time.time() - step_start)}")

    print("[Step 2/2] Predicting ΔΔG and ΔTm of candidate mutants...")

    def process_data(data):
        pair, plddt = data['pair'][0], data['plddt'][0]
        max_index = torch.argmax(plddt.mean(1))
        pair_max = pair[max_index].clone().detach().cpu().numpy()
        plddt_max = plddt[max_index].clone().detach().cpu().numpy()
        return pair_max, plddt_max

    wt_data = base_model.inference(target_sequence)
    ddG_preds, dTm_preds, plddts = [], [], []

    for i in tqdm(range(len(sele_mutants)), desc=f"Predicting"):
        mut_seq = sele_mutants[i]
        mut_data = base_model.inference(mut_seq)
        mut_pos = (wt_data['tokens'] != mut_data['tokens']).int().to(device)

        with torch.no_grad():
            ddG, dTm = stab_model(wt_data, mut_data, mut_pos)
        ddG_preds.append(ddG.item())
        dTm_preds.append(dTm.item())
        _, plddt_max = process_data(mut_data)
        plddts.append(plddt_max.mean())

    library['ddG'] = ddG_preds
    library['dTm'] = dTm_preds
    library['plddt'] = plddts
    output_csv = os.path.join(args.output, 'library.csv')
    library.to_csv(output_csv, index=False)

    print(f"\nPrediction completed. Results saved to {output_csv}")
    print_stage_end(3, start_time)

if __name__ == "__main__":
    print_section_header("START MUTANT LIBRARY CONSTRUCTION SCRIPT")
    start_total_time = time.time()

    print("--- Execution Parameters ---")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 30)

    assert os.path.exists(args.fasta), "!!! Input protein sequence does not exist !!!"
    target_name, target_sequence = read_fasta(args.fasta)
    assert os.path.exists(args.mutants), "!!! Mutation data does not exist, please run 2_directed_evolution.py first !!!"
    os.makedirs(args.output, exist_ok=True)

    if not check_env(env_name):
        print(f"Please create and install the required dependencies into the Conda environment '{env_name}' first.")
        sys.exit(1)
    
    try:
        run_first_stage()
        library = run_second_stage()
        run_third_stage(library)
    except Exception as e:
        print(f"\n\n CRITICAL ERROR during script execution: {e}")
        if ray.is_initialized():
             ray.shutdown()
        sys.exit(1)

    end_total_time = time.time()
    total_elapsed = end_total_time - start_total_time
    
    print_section_header("SCRIPT COMPLETED SUCCESSFULLY")
    print(f"All stages finished.")
    print(f"Total Execution Time: {format_time(total_elapsed)}")
    print("=" * 60)
