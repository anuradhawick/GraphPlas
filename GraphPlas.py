 
#!/usr/bin/env python3

import logging
import sys
from collections import defaultdict
import re
import argparse
import os
from tqdm import tqdm
from Bio import SeqIO
from graphplas_utils import graphplas_utils, graphplas_core

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def main(*args, **kwargs):
    contigs_path = kwargs["contigs_path"]
    contigs_paths_path = kwargs["contigs_paths_path"]
    graph_path = kwargs["graph_path"]
    threads = kwargs["threads"]
    ground_truth = kwargs["ground_truth"]
    output = kwargs["output"]
    plots = kwargs["plots"]
    truth_available = ground_truth is not None

    logger.info("Classifying the initial set of contigs longer than 1000 bp.")
    contig_prob, contig_coverage, contig_length, contig_profile = graphplas_utils.classify_using_plasclass(contigs_path, threads)
    logger.info("Classifying the initial set of contigs longer than 1000 bp complete.")

    logger.info("Computing probability thresholds.")
    probs = [v for k, v in contig_prob.items()]
    plas_prob, chrom_prob = graphplas_core.obtain_prob_thresholds(probs)
    logger.info("Computing probability thresholds complete.")

    logger.debug(f"Profile thresholds plasmids = {plas_prob} chromosomes = {chrom_prob}")
    contig_profile = graphplas_core.scale_trimer_freqs(contig_profile)

    contig_type = defaultdict(lambda: "unclassified")
    contig_class = defaultdict(lambda: "unclassified")
    contig_class.update({k: "plasmid" for k, v in contig_prob.items() if v > plas_prob})
    contig_class.update({k: "chromosome" for k, v in contig_prob.items() if v <= chrom_prob})
    contig_class.update({k: "unclassified" for k, v in contig_prob.items() if k not in contig_class})

    if truth_available:
        with open(ground_truth) as truth:
            for line in truth:
                node_id, true_label = line.strip().split("\t")
                node_id = re.sub(r'_length_[0-9]+_cov_[0-9]+(.)*$', '', node_id)
                contig_type[node_id] = true_label

    # building the graph with initial classifications
    logger.debug(f"Building graph.")
    graph = graphplas_utils.build_graph(contigs_paths_path, graph_path, contig_coverage, contig_length, contig_type, contig_class, contig_profile)    
    logger.debug(f"Building graph complete.")

    if plots:
        graph_layout = graph.layout_fruchterman_reingold()
        graphplas_utils.plot_igraph(graph, graph_layout, f"{output}/images/graph.png", "raw")
        graphplas_utils.plot_igraph(graph, graph_layout, f"{output}/images/1.png", "assigned_label")

        if truth_available:
            graphplas_utils.plot_igraph(graph, graph_layout, f"{output}/images/truth.png", "ground_truth")

    if truth_available:
        logger.info("Evaluating results of the initial classification")
        logger.info(graphplas_core.evaluate_corrected_labels(graph))

    # propagate the labels using the topology
    graph = graphplas_core.correct_using_topology(graph)

    if plots:
        graphplas_utils.plot_igraph(graph, graph_layout, f"{output}/images/2.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after topological label propagation")
        logger.info(graphplas_core.evaluate_corrected_labels(graph))

    # propagate the labels using the composition and coverage
    graph = graphplas_core.correct_using_com_cov(graph, threads)

    if plots:
        graphplas_utils.plot_igraph(graph, graph_layout, f"{output}/images/3.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after composition+coverage propagation")
        logger.info(graphplas_core.evaluate_corrected_labels(graph))

    # propagate the labels using the coverage
    graph = graphplas_core.correct_using_cov(graph, threads)

    if plots:
        graphplas_utils.plot_igraph(graph, graph_layout, f"{output}/images/4.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after coverage propagation")
        logger.info(graphplas_core.evaluate_corrected_labels(graph))  

    # label refinement using topology
    graph = graphplas_core.refine_graph_labels(graph)   

    if plots:
        graphplas_utils.plot_igraph(graph, graph_layout, f"{output}/images/5.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after coverage propagation")
        logger.info(graphplas_core.evaluate_corrected_labels(graph)) 

    logger.info(f"Writing the results to output file: {output}/final.txt") 
    with open(f"{output}/final.txt", "w+") as out_file:
        for v in tqdm(graph.vs, desc="Writing to file."):
            out_file.write(f"{v['label_raw']}\t{v['corrected_label']}\n")
    logger.info(f"Writing the results to output file: {output}/final.txt complete.") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""GraphPlas Plasmid Detection Using Assembly Graph""")

    parser.add_argument('--spades-path', '-s',
                        help="Assembly path",
                        type=str,
                        required=True)
    parser.add_argument('--output', '-o',
                        help="Contigs file (contigs.fasta)",
                        type=str,
                        required=True)                             
    parser.add_argument('--threads', '-t',
                        help="Thread limit",
                        type=int,
                        default=8,
                        required=False)
    parser.add_argument('--plots',
                    action='store_true',
                    help="Whether to plot the graph images")
    parser.add_argument('--ground-truth', '-i',
                        help="Ground truths file (tab separated file with contig_id and truth label on each line).\n Plasmid label = plasmid, Chromosome label =  chromosome",
                        type=str,
                        required=False,
                        default=None)   

    args = parser.parse_args()

    ground_truth = args.ground_truth
    spades_path = args.spades_path

    contigs_path = f"{args.spades_path}/contigs.fasta"
    contigs_paths_path = f"{args.spades_path}/contigs.paths"
    graph_path = f"{args.spades_path}/assembly_graph_with_scaffolds.gfa"

    logger = logging.getLogger('GraphPlas')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    consoleHeader = logging.StreamHandler()
    consoleHeader.setFormatter(formatter)
    consoleHeader.setLevel(logging.INFO)
    logger.addHandler(consoleHeader)

    if not os.path.isfile(contigs_path) or not os.path.isfile(contigs_paths_path) or not os.path.isfile(graph_path):
        logger.error("One of the files were not found in the assembly path")
        logger.error("Ensure you have all the files: contigs.fasta, contigs.paths and assembly_graph_with_scaffolds.gfa")
        sys.exit(1)

    output = args.output
    threads = args.threads
    plots = args.plots

    if not os.path.isdir(output):
        os.makedirs(output)
        logger.debug(f"Created directory {output}")        
        fileHandler = logging.FileHandler(f"{output}/graphplas.log")
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    else:
        fileHandler = logging.FileHandler(f"{output}/graphplas.log")
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.debug(f"Directory {output} exists. Continue without creating.")
    
    if plots and not os.path.isdir(f"{output}/images"):
        os.makedirs(f"{output}/images")
    
    main(contigs_path=contigs_path, contigs_paths_path=contigs_paths_path, graph_path=graph_path, threads=threads, ground_truth=ground_truth, output=output, plots=plots)

    logger.info("Thank you for using GraphPlas! Bye...!")

    logger.removeHandler(fileHandler)
    logger.removeHandler(consoleHeader)