import os, sys
import argparse
import time
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
from relavdep.modules.utils._format import (
    absolute_path,
    format_duration,
    print_key_values,
    print_section_header,
    print_stage_end,
    print_stage_start,
    print_step_end,
    print_step_start,
)

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
TOTAL_STAGES = 3

def check_env(env_name):
    print(f"Checking Conda environment: {env_name}")
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"],
            check=True,
            capture_output=True,
            text=True
        )

        if env_name in result.stdout:
            print(f"  -> Environment '{env_name}' is available.")
            return True
        else:
            print(f"ERROR: Conda environment '{env_name}' not found.")
            return False
    except subprocess.CalledProcessError:
        print("ERROR: Failed to execute 'conda info --envs'. Please check if Conda is working properly.")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: 'conda' command not found. Please ensure Conda is properly installed and configured in your system's PATH.")
        sys.exit(1)

def run_first_stage():
    start_time = print_stage_start(1, TOTAL_STAGES, "Extracting DHR Embeddings")
    os.makedirs(embeddings_path, exist_ok=True)

    print_key_values("Embedding Extraction", [
        ("Input mutants", absolute_path(args.mutants)),
        ("Output directory", absolute_path(embeddings_path)),
        ("Conda environment", env_name),
        ("Script", script),
    ])

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
    print("Command:")
    print("  " + " ".join(command))

    try:
        step_start = print_step_start(1, 1, "Running embedding extraction command")
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print_step_end(step_start, "Embeddings extraction completed")
        print(f"Embeddings saved under: {absolute_path(embeddings_path)}")
        print_stage_end(1, start_time)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error code: {e.returncode}")
        print("--- Stdout ---\n", e.stdout)
        print("--- Stderr ---\n", e.stderr)
        print_stage_end(1, start_time, "FAILED")
        sys.exit(1)

def run_second_stage():
    start_time = print_stage_start(2, TOTAL_STAGES, "Library Construction & Optimization")

    embeddings_file = os.path.join(embeddings_path, "embeddings.pt")
    print(f"Loading embeddings: {absolute_path(embeddings_file)}")
    raw_data = torch.load(embeddings_file)

    if args.cutoff >= raw_data['fitness'][args.size]:
        print("ERROR: Inappropriate cutoff. Please choose a lower fitness cutoff.")
        print_stage_end(2, start_time, "FAILED")
        sys.exit(1)
    if args.size < 10:
        print("ERROR: Library size cannot be less than 10.")
        print_stage_end(2, start_time, "FAILED")
        sys.exit(1)

    cutoff_index = np.where(np.array(raw_data['fitness']) > args.cutoff)[0][-1] + 1

    sele_mutants = raw_data['mutant'][:cutoff_index]
    sele_sequences = raw_data['sequence'][:cutoff_index]
    sele_embeddings = raw_data['embedding'][:cutoff_index]
    sele_fitness = raw_data['fitness'][:cutoff_index]

    data_df = pd.DataFrame({"mutant": sele_mutants, "sequence": sele_sequences, "fitness": sele_fitness})

    if args.size > len(data_df):
        print("ERROR: Library size must be less than the number of selected mutants.")
        print_stage_end(2, start_time, "FAILED")
        sys.exit(1)

    print_key_values("Selection Summary", [
        ("Reference protein", target_name),
        ("Sequence length", len(target_sequence)),
        ("Total embedded mutants", len(raw_data["mutant"])),
        ("Fitness cutoff", args.cutoff),
        ("Selected mutants", f"{len(data_df)} (fitness > cutoff)"),
        ("Target library size", args.size),
        ("Selected fitness range", f"{min(sele_fitness):.4f} to {max(sele_fitness):.4f}"),
    ])

    step_start = print_step_start(1, 5, "Projecting DHR embeddings with t-SNE")

    tsne = TSNE(n_components=2, random_state=args.seed)
    tsne_result = tsne.fit_transform(sele_embeddings)
    print_step_end(step_start, "t-SNE completed")

    step_start = print_step_start(2, 5, "Selecting best K by silhouette score (K=4..10)")
    best_k, best_score = 0, -1
    for k in range(4, 11):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=args.seed).fit(sele_embeddings)
        score = silhouette_score(sele_embeddings, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    print_step_end(step_start, f"Best K={best_k}, silhouette={best_score:.4f}")

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

    step_start = print_step_start(3, 5, f"K-means clustering with K={best_k}")
    kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=args.seed)
    clusters = kmeans.fit_predict(sele_embeddings)
    cluster_counts = Counter(clusters)
    print_step_end(step_start, f"Clustering completed across {len(cluster_counts)} clusters")

    step_start = print_step_start(4, 5, "Multi-objective optimization for fitness and diversity")
    ray.init(log_to_driver=False, _temp_dir='/tmp/ray', num_cpus=args.n_cpu)
    print(f"  -> Ray initialized with {args.n_cpu} CPU(s).")

    lambda_list = np.arange(0.01, 1.01, 0.01)
    starting_sequences, starting_fitness = init_library(clusters)
    iterations = max(len(data_df), 2000)
    print_key_values("Optimization Setup", [
        ("Lambda candidates", len(lambda_list)),
        ("Iterations per lambda", iterations),
        ("Initial mean fitness", f"{np.mean(starting_fitness):.4f}"),
        ("CPU workers", args.n_cpu),
    ])

    futures = [optimization.remote(starting_sequences, starting_fitness, lam, args.seed, iterations=iterations) for lam in lambda_list]

    sequences_history, fitness_history, diversity_history = [], [], []

    for future in tqdm(futures, desc="Optimizing lambda grid"):
        result = ray.get(future)
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

    print_step_end(step_start, f"Optimization completed; best lambda={best_lam:.2f}")

    ray.shutdown()
    print("  -> Ray shutdown.")

    step_start = print_step_start(5, 5, "Plotting figures (library.png and frequency.png)")

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
    library_plot = os.path.join(args.output, 'library.png')
    plt.savefig(library_plot, dpi=300)

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
    frequency_plot = os.path.join(args.output, 'frequency.png')
    plt.savefig(frequency_plot, dpi=300)

    print_step_end(step_start, "Figures generated")
    print_key_values("Library Summary", [
        ("Best lambda", f"{best_lam:.2f}"),
        ("Mean fitness", f"{np.mean(library_fitness):.4f}"),
        ("Diversity", f"{library_diversity:.4f}"),
        ("Library plot", absolute_path(library_plot)),
        ("Frequency plot", absolute_path(frequency_plot)),
    ])

    print_stage_end(2, start_time)

    return library

def run_third_stage(library):
    start_time = print_stage_start(3, TOTAL_STAGES, "Stability Prediction (ΔΔG & ΔTm)")

    sele_mutants = library["sequence"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_key_values("Stability Prediction Setup", [
        ("Candidate mutants", len(sele_mutants)),
        ("Device", device),
        ("Base model directory", absolute_path("models")),
        ("Stability model", absolute_path("models/SPIRED-Stab.pth")),
    ])

    step_start = print_step_start(1, 2, "Loading stability prediction models")

    base_model = BaseModel(data_dir="models", device=device)

    stab_model = Model(node_dim = 32, num_layer = 3, n_head = 8, pair_dim = 64)
    best_model = torch.load("models/SPIRED-Stab.pth", map_location=torch.device('cpu')).copy()
    best_dict = {k.split('Stab.')[-1]: v for k, v in best_model.items() if k.startswith('Stab')}
    stab_model.load_state_dict(best_dict)
    stab_model.eval().to(device)

    print_step_end(step_start, "Models loaded")

    step_start = print_step_start(2, 2, "Predicting ΔΔG and ΔTm for candidate mutants")

    def process_data(data):
        pair, plddt = data['pair'][0], data['plddt'][0]
        max_index = torch.argmax(plddt.mean(1))
        pair_max = pair[max_index].clone().detach().cpu().numpy()
        plddt_max = plddt[max_index].clone().detach().cpu().numpy()
        return pair_max, plddt_max

    wt_data = base_model.inference(target_sequence)
    ddG_preds, dTm_preds, plddts = [], [], []

    for i in tqdm(range(len(sele_mutants)), desc="Predicting stability"):
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

    print_step_end(step_start, "Stability prediction completed")
    print_key_values("Final Output", [
        ("Library CSV", absolute_path(output_csv)),
        ("Mean ddG", f"{np.mean(ddG_preds):.4f}"),
        ("Mean dTm", f"{np.mean(dTm_preds):.4f}"),
        ("Mean pLDDT", f"{np.mean(plddts):.4f}"),
    ])
    print_stage_end(3, start_time)

if __name__ == "__main__":
    print_section_header("START MUTANT LIBRARY CONSTRUCTION SCRIPT")
    start_total_time = time.time()

    print_key_values("Execution Parameters", [
        ("fasta", absolute_path(args.fasta)),
        ("mutants", absolute_path(args.mutants)),
        ("output", absolute_path(args.output)),
        ("cutoff", args.cutoff),
        ("size", args.size),
        ("seed", args.seed),
        ("n_cpu", args.n_cpu),
    ])

    assert os.path.exists(args.fasta), "!!! Input protein sequence does not exist !!!"
    target_name, target_sequence = read_fasta(args.fasta)
    assert os.path.exists(args.mutants), "!!! Mutation data does not exist, please run 2_run_directed_evolution.py first !!!"
    os.makedirs(args.output, exist_ok=True)
    print_key_values("Input Summary", [
        ("Reference protein", target_name),
        ("Sequence length", len(target_sequence)),
        ("Output directory", absolute_path(args.output)),
    ])

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
    print("All stages finished.")
    print(f"Total execution time: {format_duration(total_elapsed)}")
    print("=" * 60)
