from collections import defaultdict
import logging
import math
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import poisson
from graphplas_utils import graphplas_utils

logger = logging.getLogger('GraphPlas')

mu_intra, sigma_intra = 0, 0.01037897 / 2.
mu_inter, sigma_inter = 0.0676654, 0.03419337

def obtain_prob_thresholds(probs):
    probs = np.array(probs)
    total = len(probs)

    if np.sum(np.where(probs>0.99, 1, 0))/total > 0.05:
        return 0.99, 0.7
    return 0.7, 0.3

def distance_func(data):
    if len(data)==4:
        return data[2] * data[0] * data[1]
    else:
        return data[1] * data[0]

def majority_voting(data, votables):
    data = data[:5]
    plasmids = 0
    chromosomes = 0
    
    for row in data:
        if row[-1] == "plasmid":
            plasmids += 1
        else:
            chromosomes += 1
    
    if plasmids > chromosomes:
        return "plasmid"
    elif chromosomes > plasmids:
        return "chromosome"
    return "unclassified"

def normpdf(x, mean, sd):
    var = sd ** 2.
    denom = 2. * np.pi * var ** .5
    num = math.exp((-(x - mean) ** 2.) / (2. * var))
    
    return num / denom

def dist_com_prob(vec_1, vec_2):
    eu_distance = np.sum((vec_1 - vec_2)**2)**0.5
    prob_comp = normpdf(eu_distance, mu_intra, sigma_intra) / (normpdf(eu_distance, mu_intra, sigma_intra) + normpdf(eu_distance, mu_inter, sigma_inter))

    com_dist = 1 - prob_comp
        
    return com_dist

def dist_cov_prob(cov_1, cov_2):
    prob_cov = poisson.pmf(int(cov_1), int(cov_2))

    cov_dist = 1 - prob_cov 
    
    return cov_dist

def dist_cov_comp(X, Y):
    X_com = X[:-1]
    Y_com = Y[:-1]
    X_cov = X[-1]
    Y_cov = Y[-1]
        
    com_dist = dist_com_prob(X_com, Y_com)
    cov_dist = dist_cov_prob(X_cov, Y_cov)
                    
    return com_dist * cov_dist

def dist_cov(X, Y):
    cov_dist = dist_cov_prob(X, Y)

    return cov_dist 

def evaluate_corrected_labels(graph):
    assigned_truths = []
    assigned_labels = []

    for v in graph.vs:
        if v["ground_truth"] == "unclassified" or v["length"] < 500:
            continue

        assigned_truths.append(v["ground_truth"])
        assigned_labels.append(v["corrected_label"])

    return f"Evaluation results {graphplas_utils.evaluate(assigned_truths, assigned_labels)}"

def scale_trimer_freqs(contig_profile):
    logger.debug(f"Scaling the contig trimer profiles. No. of rofiles = {len(contig_profile)}")
    keys = [key for key in list(contig_profile.keys())]
    profiles = MinMaxScaler().fit_transform(np.array([contig_profile[key] for key in keys]))
    contig_profile = {k: p for k, p in zip(keys, profiles)}
    logger.debug(f"Scaling the contig trimer profiles completed.")

    return contig_profile

def label_prop(transition_matrix, walk_probabilities, diff, max_iter, labelled):
    itr = 0
    T = transition_matrix
    current_diff = np.inf
    Y = walk_probabilities
    Y_init = np.copy(Y)
    Y1 = Y
    
    while current_diff > diff and itr < max_iter:        
        Y0 = Y1
        Y1 = np.matmul(T, Y0)
        
        # Clamp labelled data
        for i in range(Y_init.shape[0]):
            if i in labelled:
                for j in range(Y_init.shape[1]):
                    if i==j:
                        Y1[i][j] = Y_init[i][j]
            else:
                if np.sum(Y1[i]) > 0:
                    Y1[i] = Y1[i]/np.sum(Y1[i])
        
        # Get difference between values of Y(t+1) and Y(t)
        current_diff = np.sum(np.abs(Y1-Y0))
        itr += 1
                
    return Y1

def correct_using_topology(graph):
    # set corrected label and assigned labels as same
    for v in graph.vs:
        if v["assigned_label"] != "unclassified":
            v["corrected_label"] = v["assigned_label"]

    labelled_vertices = []
    unlabelled_vertices = []

    for v in graph.vs:
        degree = len(graph.neighbors(v))

        if degree == 0:
            continue
        elif v["assigned_label"] != "unclassified":
            labelled_vertices.append(v)
        else:
            unlabelled_vertices.append(v)

    vetices_count = len(labelled_vertices) + len(unlabelled_vertices)

    index_label_map = {}
    label_index_map = {}
    degree_matrix = np.zeros(shape=(vetices_count, vetices_count))
    adjacency_matrix = np.zeros(shape=(vetices_count, vetices_count))
    walk_probabilities = np.zeros(shape=(vetices_count, len(labelled_vertices)))

    for n, v in enumerate(labelled_vertices + unlabelled_vertices):
        degree = len(graph.neighbors(v))
        degree_matrix[n][n] = degree
        index_label_map[n] = v 
        label_index_map[v["label_raw"]] = n

    for n, v in enumerate(labelled_vertices + unlabelled_vertices): 
        if v["assigned_label"] != "unclassified":
            walk_probabilities[n][n] = 1
        for ni in graph.neighbors(v):
            nv = graph.vs[ni]
            adjacency_matrix[n][label_index_map[nv["label_raw"]]] = 1

    logger.debug(degree_matrix.shape)
    logger.debug(adjacency_matrix.shape)
    logger.debug(walk_probabilities.shape)

    degree_matrix_inv = np.linalg.inv(degree_matrix)
    transition_matrix = np.matmul(degree_matrix_inv, adjacency_matrix)

    contig_label_probabilities = label_prop(transition_matrix, walk_probabilities,  0.00001, 1000, {i for i in range(len(labelled_vertices))})
    contig_topological_neighbours = { v["id"]: { labelled_vertices[i]["id"]: 1/prob for i, prob in enumerate(contig_label_probabilities[label_index_map[v["label_raw"]]]) if prob > 0 } for  v in unlabelled_vertices }

    # Correct labels using reachable labelled vertices, coverage and composition
    for v in graph.vs:
        if v["assigned_label"] == "unclassified" and len(graph.neighbors(v)) > 0 and v["length"] > 1000:
            topological_neighbours = contig_topological_neighbours[v["id"]]
            reachable_labels = set()
            reachables = []
            
            for node_id, distance in topological_neighbours.items():
                reachable_labels.add(graph.vs[node_id]["assigned_label"])
                
                coverage_distance = dist_cov_prob(graph.vs[node_id]["coverage"], v["coverage"])
                reachables_data = [coverage_distance, distance, graph.vs[node_id]["assigned_label"]]
                
                if graph.vs[node_id]["length"] > 1000:        
                    composition_distance = dist_com_prob(graph.vs[node_id]["profile"], v["profile"])
                    reachables_data = [composition_distance] + reachables_data
                    
                reachables.append(reachables_data)
            
            
            
                
            if len(reachable_labels) == 1:
                v["corrected_label"] = list(reachable_labels)[0]
            elif len(reachable_labels) == 2:
                reachables_ordered = sorted(reachables, key=distance_func)
                
                chosen_label = majority_voting(reachables_ordered, 5)
                v["corrected_label"] = chosen_label
                
        elif v["assigned_label"] == "unclassified" and len(graph.neighbors(v)) > 0:
            topological_neighbours = contig_topological_neighbours[v["id"]]
            reachable_labels = set()
            reachables = []
            
            for node_id, distance in topological_neighbours.items():
                reachable_labels.add(graph.vs[node_id]["assigned_label"])
                
                coverage_distance = dist_cov_prob(graph.vs[node_id]["coverage"], v["coverage"])
                reachables_data = [coverage_distance, distance, graph.vs[node_id]["assigned_label"]]
                    
                reachables.append(reachables_data)
            
            if len(reachable_labels) == 1:
                v["corrected_label"] = list(reachable_labels)[0]
            elif len(reachable_labels) == 2:
                reachables_ordered = sorted(reachables, key=distance_func)
                
                chosen_label = majority_voting(reachables_ordered, 5)
                v["corrected_label"] = chosen_label
    return graph

def correct_using_com_cov(graph, threads):
    logger.debug("Correction using coverage and composition.")
    to_predict_profiles = [list(v["profile"]) + [v["coverage"]] for v in graph.vs if v["corrected_label"]=="unclassified"  and v["length"] > 1000]

    # test to check if there are unclassified vertices
    if len(to_predict_profiles) > 0:
        logger.debug(f"Unclassified count = {len(to_predict_profiles)}")
        classified_profiles = [list(v["profile"]) + [v["coverage"]] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] > 1000 and len(graph.neighbors(v)) < 3]
        classified_labels = [v["corrected_label"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] > 1000 and len(graph.neighbors(v)) < 3]

        # test to check if there are classified vertices meeting the criteria
        if len(classified_labels) > 0:
            logger.debug(f"Classified count = {len(classified_labels)}")
            classified_profiles = np.array(classified_profiles)
            neigh = KNeighborsClassifier(n_neighbors=10, weights='uniform', metric=dist_cov_comp, n_jobs=threads)
            neigh.fit(classified_profiles, classified_labels)

            predictions = neigh.predict(to_predict_profiles)   

            for v, p in zip([v for v in graph.vs if v["corrected_label"]=="unclassified"  and v["length"] > 1000], predictions):
                v["corrected_label"] = p

    logger.debug("Correction using coverage and composition completed.")

    return graph

def correct_using_cov(graph, threads):
    logger.debug("Correction using coverage.")
    all_missed_coverages = np.array([v["coverage"] for v in graph.vs if v["corrected_label"] =="unclassified" and v["length"] > 500]).reshape(-1, 1)

    # test to check if there are unclassified vertices
    if len(all_missed_coverages) > 0:
        logger.debug(f"Unclassified count = {len(all_missed_coverages)}")
        # classified_coverages = [v["coverage"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] > 1000 and len(graph.neighbors(v)) < 3]
        # classified_labels = [v["corrected_label"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] > 1000 and len(graph.neighbors(v)) < 3]
        classified_coverages = [v["coverage"] for v in graph.vs if v["corrected_label"]=="plasmid" and v["length"] > 1000 and len(graph.neighbors(v)) < 3][:100]
        classified_coverages += [v["coverage"] for v in graph.vs if v["corrected_label"]=="chromosome" and v["length"] > 1000 and len(graph.neighbors(v)) < 3][:100]
        classified_labels = [v["corrected_label"] for v in graph.vs if v["corrected_label"]=="plasmid" and v["length"] > 1000 and len(graph.neighbors(v)) < 3][:100]
        classified_labels += [v["corrected_label"] for v in graph.vs if v["corrected_label"]=="chromosome" and v["length"] > 1000 and len(graph.neighbors(v)) < 3][:100]

        # test to check if there are classified vertices meeting the criteria
        if len(classified_labels) > 0:
            logger.debug(f"Classified count = {len(classified_labels)}")
            classified_coverages = np.array(classified_coverages).reshape(-1, 1)
            neigh = KNeighborsClassifier(n_neighbors=10, weights='uniform', metric=dist_cov, n_jobs=threads)
            neigh.fit(classified_coverages, classified_labels)

            predictions = neigh.predict(all_missed_coverages)   

            for v, p in zip([v for v in graph.vs if v["corrected_label"] =="unclassified"], predictions):
                v["corrected_label"] = p

    logger.debug("Correction using coverage completed.")

    return graph

def refine_graph_labels(graph):
    logger.debug("Refining labels.")
    changed = True

    while changed:
        changed = False
        for u in graph.vs:
            neighbor_ids = graph.neighbors(u)
            first_label = u["corrected_label"]

            if len(neighbor_ids) > 1:
                # assign to majority label
                chromosome_score = 0
                plasmid_score = 0

                for n in neighbor_ids:
                    if graph.vs[n]["corrected_label"] == "chromosome":
                        chromosome_score += 1
                    elif graph.vs[n]["corrected_label"] == "plasmid":
                        plasmid_score += 1

                if chromosome_score > plasmid_score:
                    u["corrected_label"] = "chromosome"
                elif chromosome_score < plasmid_score:
                    u["corrected_label"] = "plasmid"

                if first_label != u["corrected_label"]:
                    changed = True
                    
    for u in graph.vs:
        neighbor_ids = graph.neighbors(u)

        if len(neighbor_ids) == 1 and graph.vs[neighbor_ids[0]]["corrected_label"]!="unclassified":
            u["corrected_label"] = graph.vs[neighbor_ids[0]]["corrected_label"]
    
    logger.debug("Refining labels completed.")

    return graph