import argparse
import os

import numpy as np


int2A = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C',
    5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
    10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P',
    15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V'
}
A2int = {value: key for key, value in int2A.items()}


def parse_positions(raw_positions):
    if not raw_positions:
        return []
    return [int(pos) for pos in raw_positions.replace(',', ' ').split()]


def parse_mutations(raw_mutations):
    if not raw_mutations:
        return []
    return raw_mutations.replace(',', ' ').split()


def mutation_to_action(mut, use_original_residue=False):
    pos = int(mut[1:-1])
    residue = mut[0] if use_original_residue else mut[-1]
    if residue not in A2int:
        raise ValueError(f"Unsupported amino acid residue: {residue}")
    return (pos - 1) * 20 + A2int[residue] + 1


def normalize_sequence(sequence):
    sequence = ''.join(sequence.split()).upper()
    invalid_residues = sorted(set(sequence) - set(A2int))
    if invalid_residues:
        raise ValueError(f"Unsupported amino acid residues in sequence: {','.join(invalid_residues)}")
    return sequence


def prepare_constraint(illegal_pos=None, illegal_mut=None, legal_pos=None, legal_mut=None, sequence=None):
    illegal_pos = illegal_pos or []
    illegal_mut = illegal_mut or []
    legal_pos = legal_pos or []
    legal_mut = legal_mut or []
    sequence = normalize_sequence(sequence) if sequence else None

    legal, illegal = [], []

    # Non-mutable sites, using 1-based positions.
    for pos in illegal_pos:
        for res in range(20):
            action = (pos - 1) * 20 + res + 1
            illegal.append(action)

    for mut in illegal_mut:
        illegal.append(mutation_to_action(mut, use_original_residue=True))

    # Mutable sites, using 1-based positions.
    for pos in legal_pos:
        if sequence and (pos < 1 or pos > len(sequence)):
            raise ValueError(f"Position {pos} is out of range for sequence length {len(sequence)}.")
        for res in range(20):
            action = (pos - 1) * 20 + res + 1
            legal.append(action)

    for mut in legal_mut:
        legal.append(mutation_to_action(mut, use_original_residue=False))

    return sorted(set(illegal)), sorted(set(legal))


def prepare_fasta(sequence, pdb_id, output_path=None):
    sequence = normalize_sequence(sequence)

    output_path = output_path or os.path.join('relavdep', 'data', 'target_sequence', f'{pdb_id}.fasta')
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f'>{pdb_id}\n')
        f.write(f'{sequence}\n')

    return output_path


def default_npz_output(pdb_id):
    return os.path.join('relavdep', 'data', 'mutation_constraint', f'{pdb_id}.npz')


parser = argparse.ArgumentParser(description='Prepare fasta and mutation constraint npz files')
parser.add_argument('--pdb_id', type=str, required=True, help='PDB ID or target name used as the fasta header')
parser.add_argument('--sequence', type=str, required=True, help='Protein sequence used to generate the fasta file')
parser.add_argument('--fasta_output', type=str, default='', help='Output fasta file path. Default: relavdep/data/target_sequence/PDB_ID.fasta')
parser.add_argument('--npz_output', type=str, default='', help='Output .npz constraint file path. Default: relavdep/data/mutation_constraint/PDB_ID.npz')
parser.add_argument('--illegal_pos', type=str, default='', help='Non-mutable positions, 1-based. Example: "10,25,36"')
parser.add_argument('--illegal_mut', type=str, default='', help='Forbidden mutations. Example: "A10V,G25D"')
parser.add_argument('--legal_pos', type=str, default='', help='Mutable positions, 1-based. Example: "10,25,36"')
parser.add_argument('--legal_mut', type=str, default='', help='Allowed mutations. Example: "A10V,G25D"')
args = parser.parse_args()


if __name__ == '__main__':
    fasta_path = prepare_fasta(args.sequence, args.pdb_id, args.fasta_output or None)
    print(f"Fasta file saved to: {fasta_path}")

    should_prepare_constraint = any([args.npz_output, args.illegal_pos, args.illegal_mut, args.legal_pos, args.legal_mut])
    if should_prepare_constraint:
        npz_output = args.npz_output or default_npz_output(args.pdb_id)

        illegal, legal = prepare_constraint(
            illegal_pos=parse_positions(args.illegal_pos),
            illegal_mut=parse_mutations(args.illegal_mut),
            legal_pos=parse_positions(args.legal_pos),
            legal_mut=parse_mutations(args.legal_mut),
            sequence=args.sequence or None,
        )

        output_dir = os.path.dirname(npz_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.savez(npz_output, illegal=illegal, legal=legal)
        print(f"Constraint file saved to: {npz_output}")
        print(f"Illegal actions: {len(illegal)}")
        print(f"Legal actions: {len(legal)}")
