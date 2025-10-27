int2A = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 
         7: 'G', 8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 
         14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V'}

int2AA = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'GLU', 4: 'ASP', 5: 'PHE', 6: 'ILE', 
          7: 'HIS', 8: 'LYS', 9: 'MET', 10: 'LEU', 11: 'ASN', 12: 'GLN', 13: 'PRO', 
          14: 'SER', 15: 'ARG', 16: 'THR', 17: 'TRP', 18: 'VAL', 19: 'TYR'}

A2int = {value: key for key, value in int2A.items()}
AA2int = {value: key for key, value in int2AA.items()}

aa_list = list("ARNDCQEGHILKMFPSTWYV")