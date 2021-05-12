import itertools
import re
from collections import defaultdict
from multiprocessing import Pool
import logging
from tqdm import tqdm
from igraph import *
import numpy as np
from sklearn.metrics import precision_score, recall_score
from Bio import SeqIO
from tabulate import tabulate
import csv

complements = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
nt_bits = {'A':0,'C':1,'G':2,'T':3}
graph_colors = { "plasmid": "green", "chromosome": "orange", "unclassified": "white" }

logger = logging.getLogger('GraphPlas')

def get_rc(seq):
    rev = reversed(seq)
    return "".join([complements.get(i,i) for i in rev])

def mer2bits(kmer):
    bit_mer=nt_bits.get(kmer[0], 0)
    for c in kmer[1:]:
        bit_mer = (bit_mer << 2) | nt_bits.get(c, 0)
    return bit_mer

def compute_kmer_inds(k):
    kmer_inds = {}
    kmer_count_len = 0

    alphabet = 'ACGT'
    
    all_kmers = [''.join(kmer) for kmer in itertools.product(alphabet,repeat=k)]
    all_kmers.sort()
    ind = 0
    for kmer in all_kmers:
        bit_mer = mer2bits(kmer)
        rc_bit_mer = mer2bits(get_rc(kmer))
        if rc_bit_mer in kmer_inds:
            kmer_inds[bit_mer] = kmer_inds[rc_bit_mer]
        else:
            kmer_inds[bit_mer] = ind
            kmer_count_len += 1
            ind += 1
            
    return kmer_inds, kmer_count_len

def count_kmers(args):
    seq, k, kmer_inds, kmer_count_len = args
    profile = np.zeros(kmer_count_len)
    arrs = []
    seq = list(seq.strip())
    
    for i in range(0, len(seq) - k + 1):
        bit_mer = mer2bits(seq[i:i+k])
        index = kmer_inds[bit_mer]
        profile[index] += 1
    profile = profile/max(1, sum(profile))
    
    return profile

def evaluate(truth, clusters):
    truth = list(map(lambda x: x.lower(), truth))
    clusters = list(map(lambda x: x.lower(), clusters))
    
    clusters = [clusters[i] for i in range(len(truth)) if truth[i] != "unclassified"]
    truth = [truth[i] for i in range(len(truth)) if truth[i] != "unclassified"]
    
    truth_count_plasmid = truth.count("plasmid")
    truth_count_chromosome = truth.count("chromosome")
    
    string_output  = f"\nPlasmids    {truth_count_plasmid}"
    string_output += f"\nChromosomes {truth_count_chromosome}\n"
    
    truth = np.array(truth)
    clusters = np.array(clusters)

    classified_truths = truth[clusters!='unclassified']
    classified_clusters = clusters[clusters!='unclassified']
        
    c_as_c = 0
    c_as_p = 0
    p_as_p = 0
    p_as_c = 0
    c_as_u = 0
    p_as_u = 0
    
    for t, c in zip(truth, clusters):
        if c==t=='plasmid':
            p_as_p+=1
        elif c==t=='chromosome':
            c_as_c+=1
        elif t.lower()=='chromosome':
            if c.lower()=='plasmid':
                c_as_p+=1
            else:
                c_as_u+=1
        else:
            if c.lower()=='chromosome':
                p_as_c+=1
            else:
                p_as_u+=1
    string_output += tabulate([["Chromosome",c_as_c, c_as_p, c_as_u],
                    ["Plasmid",p_as_c, p_as_p, p_as_u]], 
                   headers=["", "Chromosome", "Plasmid", "Unclassified"], tablefmt="fancy_grid")
    
    precision = 100 * precision_score(classified_truths, classified_clusters, average='macro')
    recall = 100 * recall_score(truth, clusters, average='macro')
    f1 = precision * recall * 2 / (precision+recall)

    string_output += f"\n\nPrecision {precision:3.2f}"
    string_output += f"\nRecall    {recall:3.2f}"
    string_output += f"\nF1        {f1:3.2f}"
    plasmid_precision = (p_as_p)/max(p_as_p+c_as_p, 1)
    chromosome_precision = (c_as_c)/max(c_as_c+p_as_c, 1)
    
    plasmid_recall = (p_as_p)/max(p_as_p+p_as_c+p_as_u, 1)
    chromosome_recall = (c_as_c)/max(c_as_c+c_as_p+c_as_u, 1)
    
    string_output += f"\n\nPlasmid precision    {100 * plasmid_precision:3.2f}"
    string_output += f"\nChromosome precision {100 * chromosome_precision:3.2f}"
    string_output += f"\nPlasmid recall       {100 * plasmid_recall:3.2f}"
    string_output += f"\nChromosome recall    {100 * chromosome_recall:3.2f}"

    return string_output    
    
def build_graph(contigs_paths_path, graph_path, contig_coverage, contig_length, contig_type, contig_class, contig_profile):
    paths = {}
    segment_contigs = {}
    node_count = 0

    # Get contig paths from contigs.paths
    with open(contigs_paths_path) as file:
        name = file.readline()
        cpath = file.readline()

        while name != "" and cpath != "":

            while ";" in cpath:
                cpath = cpath[:-2]+","+file.readline()

            start = 'NODE_'
            end = '_length_'
            contig_num = str(int(re.search('%s(.*)%s' % (start, end), name).group(1))-1)

            segments = cpath.rstrip().split(",")

            if contig_num not in paths:
                node_count += 1
                paths[contig_num] = [segments[0], segments[-1]]

            segment = segments[0]
            if segment not in segment_contigs:
                segment_contigs[segment] = set([contig_num])
            else:
                segment_contigs[segment].add(contig_num)

            segment = segments[-1]
            if segment not in segment_contigs:
                segment_contigs[segment] = set([contig_num])
            else:
                segment_contigs[segment].add(contig_num)

            name = file.readline()
            cpath = file.readline()

    links = []
    links_map = defaultdict(set)

    # Get contig paths from contigs.paths
    with open(graph_path) as file:
        for line in file:
            if line[0] == "L" :
                strings = line.strip().split("\t")
                strings[1] = re.sub(r'^EDGE_', '', strings[1])
                strings[1] = re.sub(r'_length(.)*', '', strings[1])
                strings[3] = re.sub(r'^EDGE_', '', strings[3])
                strings[3] = re.sub(r'_length(.)*', '', strings[3])
                f1, f2 = strings[1]+strings[2], strings[3]+strings[4]
                links_map[f1].add(f2)
                links_map[f2].add(f1)
                links.append(strings[1]+strings[2]+" "+strings[3]+strings[4])

    g = Graph()
    g.add_vertices(node_count)

    edges = []

    for i in range(len(g.vs)):
        node_label = f"NODE_{i+1}"
        g.vs[i]["id"] = i
        g.vs[i]["label"] = f"NODE_{i+1}" + "\n" + str(contig_coverage[f"NODE_{i+1}"]) + "\n" + str(contig_length[f"NODE_{i+1}"])
        g.vs[i]["label_raw"] = f"NODE_{i+1}"
        g.vs[i]["coverage"] = contig_coverage[node_label]
        g.vs[i]["length"] = contig_length[node_label]
        g.vs[i]["ground_truth"] = contig_type[node_label]
        g.vs[i]["assigned_label"] = contig_class[node_label]
        g.vs[i]["corrected_label"] = contig_class[node_label]

        if node_label in contig_profile:
            g.vs[i]["profile"] = contig_profile[node_label]

    for i in range(len(paths)):
        segments = paths[str(i)]

        start = segments[0]
        start_rev = ""
        if start.endswith("+"):
            start_rev = start[:-1]+"-"
        else:
            start_rev = start[:-1]+"+"

        end = segments[1]
        end_rev = ""
        if end.endswith("+"):
            end_rev = end[:-1]+"-"
        else:
            end_rev = end[:-1]+"+"

        new_links = []

        if start in links_map:
            new_links.extend(list(links_map[start]))
        if start_rev in links_map:
            new_links.extend(list(links_map[start_rev]))
        if end in links_map:
            new_links.extend(list(links_map[end]))
        if end_rev in links_map:
            new_links.extend(list(links_map[end_rev]))

        for new_link in new_links:
            if new_link in segment_contigs:
                for contig in segment_contigs[new_link]:
                    if i!=int(contig):
                        edges.append((i,int(contig)))

    g.add_edges(edges)
    g.simplify(multiple=True, loops=False, combine_edges=None)

    return g

def plot_igraph(graph, my_layout, fig_name, plot_scheme):
    logger.debug(f"Plotting image to path = {fig_name}, scheme = {plot_scheme}")

    for v in graph.vs:
        if plot_scheme == "raw":
            v["color"] = "white"
        else:    
            v["color"] = graph_colors[v[plot_scheme]]

    visual_style = {}
    visual_style["bbox"] = (4000,4000)
    visual_style["margin"] = 57
    visual_style["vertex_size"] = 75
    visual_style["vertex_label_size"] = 8
    visual_style["edge_curved"] = False
    visual_style["layout"] = my_layout

    plot(graph, fig_name, **visual_style) 

def read_pc(classification_file):
    contig_prob = {}

    for line in open(classification_file):
        contig_id, prob = line.strip().split("\t")
        contig_length = int(contig_id.split("_")[3])
        prob = float(prob)
        contig_id = re.sub(r'_length_[0-9]+_cov_[0-9]+(.)*$', '', contig_id)
        
        if contig_length >= 1000:
            contig_prob[contig_id] = prob

    return contig_prob

def read_pf(classification_file):
    contig_prob = {}

    with open(classification_file) as pf_file:
        reader = csv.reader(pf_file, delimiter="\t", quotechar='"')
        rows = list(reader)

        for row in rows[1:]:
            contig_id, prob = row[2], sum(map(float, row[24:]))
            contig_length = int(contig_id.split("_")[3])
            contig_id = re.sub(r'_length_[0-9]+_cov_[0-9]+(.)*$', '', contig_id)
            
            if contig_length >= 1000:
                contig_prob[contig_id] = prob

    return contig_prob

def classify_contigs(contigs_path, classifier, classification_file, threads):
    seq_ids = []
    seqs = []
    
    contig_coverage = {}
    contig_length = {}
    contig_profile = {}
    kmer_inds_4, kmer_count_len_4 = compute_kmer_inds(4)
    seq_count = 0
    seq_count_1000bp = 0

    for record in SeqIO.parse(contigs_path, "fasta"):
        seq_count += 1
        if len(record.seq) >= 1000:
            seq_count_1000bp += 1
    
    logger.debug(f"Total number of contigs = {seq_count}, contigs longer than 1000 bp = {seq_count_1000bp}.")

    for record in tqdm(SeqIO.parse(contigs_path, "fasta"), total=seq_count, desc="Computing plasmid probabilities."):
        seq_id = re.sub(r'_length_[0-9]+_cov_[0-9]+(.)*$', '', record.id)
        contig_coverage[seq_id] = float(record.id.split("_")[5])
        contig_length[seq_id] = len(record.seq)

        if len(record.seq) < 1000:
            continue

        seq_ids.append(seq_id)
        seqs.append(str(record.seq))

        if len(seqs) == 1000:         
            pool = Pool(8)
            record_trimers = pool.map(count_kmers, [(seq, 4, kmer_inds_4, kmer_count_len_4) for seq in seqs])
            pool.close()

            for n, profile in enumerate(record_trimers):
                contig_profile[seq_ids[n]] = profile

            seqs = []
            seq_ids = []

    if len(seqs) > 0:
        pool = Pool(8)
        record_trimers = pool.map(count_kmers, [(seq, 4, kmer_inds_4, kmer_count_len_4) for seq in seqs])
        pool.close()

        for n, profile in enumerate(record_trimers):
            contig_profile[seq_ids[n]] = profile
    if classifier == 'pc':
        return read_pc(classification_file), contig_coverage, contig_length, contig_profile
    else:
        return read_pf(classification_file), contig_coverage, contig_length, contig_profile

