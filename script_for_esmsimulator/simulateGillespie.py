import argparse
from Bio.Data import CodonTable
import numpy as np
import pandas as pd
from ete3 import Tree

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

import torch
from tqdm import tqdm


NUCLEIC_BASE_LIST = ('A','C','G','T')
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
stop_codons = standard_table.stop_codons                  
start_codons = standard_table.start_codons
# usage:
# python simulateGillespie.py --exchangeabilities ../data/coevolution.txt --eqfreq ../data/coevolution_stationary.txt --tree ../data/200k500gaps100007_20_tips.nwk --seqlen 10 --output ../output
def read_tree_from_file(file):
    with open(file, 'r') as f:
        line = ""
        for l in f:
            line += l.strip()
        t = Tree( line, format=1 )
    return t

# LG equilbrium frequencies, from http://www.atgc-montpellier.fr/download/datasets/models/lg_LG.PAML.txt
all_k_aa_proba = [0.079066, 0.055941, 0.041977, 0.053052, 0.012937, 0.040767, 0.071586, 0.057337, 
                  0.022355, 0.062157, 0.099081, 0.064600, 0.022951, 0.042302, 0.044040, 0.061197, 
                  0.053287, 0.012066, 0.034155, 0.069147 ]
# all_k_aa_proba = [0.9] + [0.1/19]*19 # For basic testing
aas = ['K', 'N', 'T', 'R', 'S', 'I', 'M', 'Q', 
       'H', 'P', 'L', 'E', 'D', 'A', 'G', 'V', 
       'Y', 'C', 'W', 'F']

aa_to_proba = dict()
i=0
for aa in aas:
    aa_to_proba[aa] = all_k_aa_proba[i]
    i = i+1
print(aa_to_proba)

def computeScale(subst_mat, eq_freq):
    diago = np.diag(subst_mat)
    tmp = np.multiply(diago, eq_freq)
    scale = -sum(tmp)
    return scale

def simulateSiteAlongBranch(rate_matrix, starting_state_int, branch_length):
    current_time = 0.0
    current_state_int = starting_state_int
    while current_time < branch_length:
        rate = rate_matrix[current_state_int,current_state_int]
        waiting_time = np.random.exponential(-1/rate)
        current_time = current_time + waiting_time
        if current_time <= branch_length:
            # print("Substitution at time "+ str(current_time))
            vec = rate_matrix[current_state_int,].flatten()
            vec[current_state_int] = 0.0
            vec = vec/sum(vec)
            current_state_int = np.random.choice(400, 1, p=vec)
            # print("New state : " + str(current_state_int))
    return(current_state_int)


def sequencesToFasta(sequences, str_states, tree):
    num_seqs = sequences.shape[0]
    seq_len = sequences.shape[1]    
    leaves = list()
    sequences_str = list()
    for leaf in tree:
        leaves.append(leaf)
        seq = sequences[leaf.id,]
        seq_str = ">" + leaf.name + "\n"
        for i in seq:
            seq_str += str_states[i]
        seq_str += "\n"
        sequences_str.append(seq_str)
    return sequences_str

def computeProbabilitiesFromLine(subst_mat, current_nt):
    subst_mat_line = subst_mat[current_nt]
    probs = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        if i != current_nt:
            probs[i] = - subst_mat_line[i] / subst_mat_line[current_nt]
    return probs

def is_stop(codon):
    codon_nt = str()
    for i in codon:
        if i==0:
            codon_nt = codon_nt+"A"
        elif i==1:
            codon_nt = codon_nt+"C"
        elif i==2:
            codon_nt = codon_nt+"G"
        elif i==3:
            codon_nt = codon_nt+"T"
        else:
            print("Error in is_stop")
            exit(-1)
    if codon_nt == stop_codons[0]:
        return True
    elif codon_nt == stop_codons[1]:
        return True
    elif codon_nt == stop_codons[2]:
        return True
    else :
        return False


def codon_to_ints(codon):
    return([eval(i) for i in [*codon.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")]])


def int_to_aa(codon_int):
    codon = NUCLEIC_BASE_LIST[codon_int[0]] + NUCLEIC_BASE_LIST[codon_int[1]] + NUCLEIC_BASE_LIST[codon_int[2]]
    return standard_table.forward_table[codon]


def translateCodonSequence(cd_seq_int):
    aa_seq = ""
    for i in range(0, len(cd_seq_int)-3, 3): # we remove the stop codon at the end
        aa_seq = aa_seq + int_to_aa(cd_seq_int[i:i+3])
    return aa_seq


def computeProbaForAllMutants(alphabet, model, sequence):
    data = [
        ("current_sequence", translateCodonSequence(sequence)),
    ]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # scoring_strategy: "masked-marginals":
    all_token_probs = []
    for i in tqdm(range(batch_tokens.size(1))):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens_masked.cuda())["logits"], dim=-1
            )
        all_token_probs.append(token_probs[:, i])  # vocab size
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    return token_probs


def main(args):
    seq_len = args.seqlen
    starting_sequence = list()
    
    if args.inputseq : 

        if len(args.inputseq) % 3 != 0 :
            print("Error : the input sequence must have a length divisible by 3")
            exit(-1)
        else :
            seq_len = int(len(args.inputseq)/3)
        
        if any(base not in NUCLEIC_BASE_LIST for base in args.inputseq):
            print("Error : the input sequence must contain only nucleotides A,C,G,T")
            exit(-1)    

        for i in range(0, len(args.inputseq), 3):
            codon = args.inputseq[i:i+3]
            starting_sequence.append(codon)

    else : 
        starting_codon_int = np.random.choice(3, None, p=np.array([1/3, 1/3, 1/3]))
        starting_sequence.append(standard_table.start_codons[starting_codon_int])
        starting_sequence_int = np.random.choice(61, seq_len-2, p=[1/61]*61)
        for i in starting_sequence_int:
            starting_sequence.append(list(standard_table.forward_table.keys())[i])
        stop_codon_int = np.random.choice(3, None, p=np.array([1/3, 1/3, 1/3]))
        starting_sequence.append(standard_table.stop_codons[stop_codon_int])
    
    starting_sequence_joined = "".join(starting_sequence)
    print(starting_sequence_joined)
    
    starting_sequence_int = [eval(i) for i in [*starting_sequence_joined.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")]]
    print(starting_sequence_int)

    #exchangeabilities_file=args.exchangeabilities
    tree_file = args.tree
    out_file = args.output
    #exch = pd.read_table(exchangeabilities_file, index_col=0)
    a= 1.0
    b= 1.0
    c= 1.0
    d= 1.0
    e= 1.0
    f= 1.0
    pi_a = 0.45 
    pi_c = 0.15
    pi_g = 0.10
    pi_t = 1.0-(pi_a+pi_c+pi_g)
    exch = [ [-(a*pi_c+b*pi_g+c*pi_t), a*pi_c, b*pi_g, c*pi_t], 
             [a*pi_a, -(a*pi_a+d*pi_g+e*pi_t), d*pi_g, e*pi_t],
             [b*pi_a, d*pi_c, -(b*pi_a+d*pi_c+f*pi_t)*pi_g, f*pi_t],
             [c*pi_a, e*pi_c, f*pi_g, -(c*pi_a+e*pi_c+f*pi_g)]]
    exch_np = np.array(exch)
    eq_freq = [pi_a, pi_c, pi_g, pi_t]
    eq_freq_np = np.array(eq_freq)
    # Computing the complete rate matrix:
    subst_mat = np.multiply(exch_np, eq_freq_np)
    # print(subst_mat.shape)
    # recompute the diagonal to make sure it is equal to minus the sum of the other terms
    subst_mat = subst_mat - np.diag(subst_mat)
    diago = -subst_mat.sum(1)
    np.fill_diagonal(subst_mat, diago)
    scale = computeScale(subst_mat, eq_freq_np)
    print("Before rescaling: "+str(scale))
    # rescaling:
    subst_mat = np.multiply(1/scale, subst_mat)
    scale = computeScale(subst_mat, eq_freq_np)
    print("After rescaling: "+str(scale))
    print(subst_mat)
    # print(subst_mat[0,].sum())
    # read tree
    tree = read_tree_from_file(tree_file)
    # print(tree)
    id=0
    for node in tree.traverse("preorder"):
        node.add_features(id=id)
        id = id + 1
    # then simulating for each site of the starting sequence
    numseq=id
    sequences = np.ndarray(shape=(numseq, 3*seq_len), dtype=int)
    np.insert(sequences, 0, np.array(starting_sequence_int), axis=0)
    sequences[0] = starting_sequence_int
    # Simulating along the tree:
    # Use the branch length to generate simulations across the sequence:
    # - draw a Poisson number num_subst of amino acid substitutions
    # - draw positions for nucleotide substitutions
    # - for each substituted position:
    #       - draw arriving nucleotide 
    #       - if it creates a stop codon : try another one
    #       - if it does not change the aa, accept
    #       - if it changes the amino acid, compare the probabilities and draw a random number to choose accordingly
    #       - do that until we have made the right number of substitutions num_subst

    if args.useesm:
        print("\n\n\t\tUsing ESM\n\n")
        model, alphabet = pretrained.load_model_and_alphabet(args.model_location[0])
        model.eval()
        if torch.cuda.is_available() :
            model = model.cuda()
            print("Transferred model to GPU")

        # batch_converter = alphabet.get_batch_converter()

        # data = [
        #     ("current_sequence", translateCodonSequence(starting_sequence_int)),
        # ]
        # batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # # scoring_strategy: "masked-marginals":
        # all_token_probs = []
        # for i in tqdm(range(batch_tokens.size(1))):
        #     batch_tokens_masked = batch_tokens.clone()
        #     batch_tokens_masked[0, i] = alphabet.mask_idx
        #     with torch.no_grad():
        #         token_probs = torch.log_softmax(
        #             model(batch_tokens_masked.cuda())["logits"], dim=-1
        #         )
        #     all_token_probs.append(token_probs[:, i])  # vocab size
        # token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        token_probs = computeProbaForAllMutants(alphabet, model, starting_sequence_int)

    for node in tree.traverse("preorder"):
            if node.is_root():
                pass
            else:
                sequences[node.id] = sequences[node.up.id]
                # - draw a Poisson number of Amino acid substitutions
                num_subst = np.random.poisson(lam=node.dist * args.rescale * seq_len)
                # print("num_subst : {}", num_subst)
                num_subst_effective = 0
                # - draw positions for these substitutions, except for the start and stop
                if num_subst >0:
                    syn= 0
                    nonsyn=0
                    current_sequence = sequences[node.up.id]
                    while num_subst_effective < num_subst:
                        site = np.random.choice(seq_len-2, 1)[0]+1
                        # - draw arriving nucleotide until arriving nucleotide does not create a stop codon
                        current_codon =  [current_sequence.item(site*3), current_sequence.item(site*3+1), current_sequence.item(site*3+2)]
                        mutated_position = np.random.choice(3, None)
                        current_nt = current_codon[mutated_position]
                        # print("current_nt: {}", current_nt)
                        probabilities = computeProbabilitiesFromLine(subst_mat, current_nt)
                        # print("probabilities: {}", probabilities)
                        new_codon = current_codon.copy()
                        new_nt = np.random.choice(4, None, p=probabilities)
                        # print("new_nt: {}", new_nt)
                        new_codon[mutated_position] = new_nt
                        # print("current_codon: {}", current_codon)
                        # print("new_codon: {}", new_codon)
                        if not is_stop(new_codon):
                            # Accept or reject based on amino acid probabilities
                            oldAA = int_to_aa(current_codon)
                            newAA = int_to_aa(new_codon)
                            if oldAA == newAA:
                                syn = syn+1
                                current_sequence.itemset(site*3, new_codon[0])
                                current_sequence.itemset(site*3+1, new_codon[1])
                                current_sequence.itemset(site*3+2, new_codon[2])
                                # num_subst_effective += 1
                            else:
                                old_prob = aa_to_proba[oldAA]
                                new_prob = aa_to_proba[newAA]
                                if args.useesm:
                                    # print("Using ESM")
                                    oldAA_encoded, newAA_encoded = alphabet.get_idx(oldAA), alphabet.get_idx(newAA)
                                    # add 1 for BOS
                                    score = token_probs[0, 1 + site, newAA_encoded] - token_probs[0, 1 + site, oldAA_encoded]
                                    old_prob = torch.exp(token_probs[0, 1 + site, newAA_encoded]).cpu().numpy()
                                    new_prob = torch.exp(token_probs[0, 1 + site, oldAA_encoded]).cpu().numpy()
                                    # print("WT: " + str(old_prob) + " ; Mutated: " + str(new_prob))
                                p = np.array([old_prob, new_prob])
                                p /= p.sum()  # normalize
                                choice = np.random.choice(2, p=p)
                                if ( choice == 1 ):
                                    nonsyn = nonsyn+1
                                    current_sequence.itemset(site*3, new_codon[0])
                                    current_sequence.itemset(site*3+1, new_codon[1])
                                    current_sequence.itemset(site*3+2, new_codon[2])
                                    num_subst_effective += 1
                                    if args.useesm: # update the probabilities from ESM
                                        token_probs = computeProbaForAllMutants(alphabet, model, current_sequence)
                    sequences[node.id] = current_sequence
                                        
                    print("SYN: {} ; NONSYN: {}", syn, nonsyn)
    seq_str = sequencesToFasta(sequences, NUCLEIC_BASE_LIST, tree)
    with open (out_file, "w") as fout:
        for s in seq_str:
            fout.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--exchangeabilities', 
                        type=str, 
                        help='/path/ to the file containing the exchangeabilities')
    parser.add_argument("--useesm", 
                        action="store_true", 
					    help="should we use ESM to accept or reject mutations?") 
    parser.add_argument("--model-location",
                        type=str,
                        help="PyTorch model file OR name of pretrained model to download (see README for models)",
                        nargs="+")
    parser.add_argument('--tree', 
                        type=str, 
                        help='/path/ to the tree file')
    parser.add_argument('--seqlen', 
                        type=int, 
                        help='number of sites to simulate')
    parser.add_argument('--rescale', 
                        type=float, 
                        default=1.0,
                        help='by how much to rescale branch lengths')
    parser.add_argument('--output', 
                        type=str, 
                        help='/path/ to output directory where the alignment will be saved') 
    parser.add_argument('--inputseq', '-i', 
                        type = str,
                        help='Input sequence composed of A, C, G, T')

    args=parser.parse_args()

    main(args) 
