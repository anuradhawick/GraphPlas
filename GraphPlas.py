#!/usr/bin/env python3

import logging
import sys
from collections import defaultdict
import re
import argparse
import os
from tqdm import tqdm
from Bio import SeqIO
from graphplas_utils import utils, core

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger('GraphPlas')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHeader = logging.StreamHandler()
consoleHeader.setFormatter(formatter)
consoleHeader.setLevel(logging.INFO)
logger.addHandler(consoleHeader)
fileHandler = None


def main(*args, **kwargs):
    contigs_path = kwargs["contigs_path"]
    contigs_paths_path = kwargs["contigs_paths_path"]
    graph_path = kwargs["graph_path"]
    threads = kwargs["threads"]
    ground_truth = kwargs["ground_truth"]
    output = kwargs["output"]
    plots = kwargs["plots"]
    classifier = kwargs["classifier"]
    classification_file = kwargs["classification_file"]
    min_contig_length = max(0, kwargs["min_contig_length"])
    truth_available = ground_truth is not None

    logger.info("Classifying the initial set of contigs longer than 1000 bp.")
    contig_prob, contig_coverage, contig_length, contig_profile = utils.classify_contigs(contigs_path, classifier, classification_file, threads)
    logger.debug("Classifying the initial set of contigs longer than 1000 bp complete.")

    logger.info("Computing probability thresholds.")
    probs = [v for k, v in contig_prob.items()]
    plas_prob, chrom_prob = core.obtain_prob_thresholds(probs, classifier)
    logger.debug("Computing probability thresholds complete.")

    logger.debug(f"Profile thresholds plasmids = {plas_prob} chromosomes = {chrom_prob}")

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

    # scaling the kmer freq vectors
    # contig_profile = core.scale_freqs(contig_profile)

    # building the graph with initial classifications
    logger.info(f"Building graph.")
    graph = utils.build_graph(contigs_paths_path, graph_path, contig_coverage, contig_length, contig_type, contig_class, contig_profile)    
    logger.debug(f"Building graph complete.")

    if plots:
        graph_layout = graph.layout_fruchterman_reingold()
        utils.plot_igraph(graph, graph_layout, f"{output}/images/graph.png", "raw")
        utils.plot_igraph(graph, graph_layout, f"{output}/images/1.png", "assigned_label")

        if truth_available:
            utils.plot_igraph(graph, graph_layout, f"{output}/images/truth.png", "ground_truth")

    if truth_available:
        logger.info("Evaluating results of the initial classification")
        logger.info(core.evaluate_corrected_labels(graph))

    # propagate the labels using the topology
    logger.info(f"Starting topological label correction.")
    graph = core.correct_using_topology(graph)
    logger.debug(f"Topological label correction complete.")

    if plots:
        utils.plot_igraph(graph, graph_layout, f"{output}/images/2.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after topological label propagation")
        logger.info(core.evaluate_corrected_labels(graph))

    # propagate the labels using the composition and coverage
    graph = core.correct_using_com_cov(graph, threads)

    if plots:
        utils.plot_igraph(graph, graph_layout, f"{output}/images/3.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after composition+coverage propagation")
        logger.info(core.evaluate_corrected_labels(graph))

    # propagate the labels using the coverage
    graph = core.correct_using_cov(graph, min_contig_length, threads)

    if plots:
        utils.plot_igraph(graph, graph_layout, f"{output}/images/4.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after coverage propagation")
        logger.info(core.evaluate_corrected_labels(graph))  

    # label refinement using topology
    graph = core.refine_graph_labels(graph)   

    if plots:
        utils.plot_igraph(graph, graph_layout, f"{output}/images/5.png", "corrected_label")

    if truth_available:
        logger.info("Evaluating results of the after coverage propagation")
        logger.info(core.evaluate_corrected_labels(graph)) 

    logger.info(f"Writing the results to output file: {output}/final.txt") 
    plasmids_count = 0
    chromosomes_count = 0
    unclassified_count = 0

    with open(f"{output}/final.txt", "w+") as out_file:
        for v in tqdm(graph.vs, desc="Writing to file."):
            if v['corrected_label'] == "plasmid":
                plasmids_count += 1
            elif v['corrected_label'] == "chromosome":
                chromosomes_count += 1
            else:
                unclassified_count += 1
            out_file.write(f"{v['label_raw']}\t{v['corrected_label']}\n")
    logger.info(f"Writing the results to output file: {output}/final.txt complete.") 
    logger.info(f"Plasmids      {plasmids_count:10}") 
    logger.info(f"Chromosomes   {chromosomes_count:10}") 
    logger.info(f"Unclassified  {unclassified_count:10}") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""GraphPlas Plasmid Detection Using Assembly Graph""", prog="GraphPlas")

    parser.add_argument('--spades-path', '-s',
                        help="Assembly path",
                        type=str,
                        required=True)
    parser.add_argument('--classifier', '-c',
                        help="Classifier [pc or pf] default=pc",
                        required=True,
                        type=str)
    parser.add_argument('--classification-file', '-f',
                        help="Path to classification file",
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
    parser.add_argument('--min_contig_length', '-m',
                        help="Minimum length of contigs to consider",
                        type=int,
                        default=500,
                        required=False)
    parser.add_argument('--plots',
                    action='store_true',
                    help="Whether to plot the graph images")
    parser.add_argument('--version', '-v',
                    action='version',
                    help="Show version.",
                    version='%(prog)s 0.1-rc')                    
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
    classifier = str(args.classifier).lower()
    classification_file = args.classification_file

    if not os.path.isfile(contigs_path) or not os.path.isfile(contigs_paths_path) or not os.path.isfile(graph_path):
        logger.error("One of the files were not found in the assembly path")
        logger.error("Ensure you have all the files: contigs.fasta, contigs.paths and assembly_graph_with_scaffolds.gfa")
        sys.exit(1)

    if not os.path.isfile(classification_file):
        logger.error("Could not file the initial classification file")
        sys.exit(1)

    if not classifier in ['pf', 'pc']:
        logger.error("Classifier must either be 'pc' or 'pf'")
        sys.exit(1)

    output = args.output
    threads = args.threads
    plots = args.plots
    min_contig_length = args.min_contig_length

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
    
    logger.info("Running command: " + " ".join(sys.argv))
    main(min_contig_length=min_contig_length, contigs_path=contigs_path, contigs_paths_path=contigs_paths_path, graph_path=graph_path, threads=threads, ground_truth=ground_truth, output=output, plots=plots, classifier=classifier, classification_file=classification_file)

    logger.info("Thank you for using GraphPlas! Bye...!")

    logger.removeHandler(fileHandler)
    logger.removeHandler(consoleHeader)