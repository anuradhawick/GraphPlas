from collections import defaultdict
import logging
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from graphplas_utils import graphplas_utils

logger = logging.getLogger('GraphPlas')

def obtain_prob_thresholds(probs):
    probs = np.array(probs)
    total = len(probs)

    if np.sum(np.where(probs>0.99, 1, 0))/total > 0.05:
        return 0.99, 0.7
    return 0.7, 0.3

def distance_func(data):
    if len(data)==4:
        return data[2] * (1 + data[0]) * (1 +  data[1])
    else:
        return data[1] * (1 + data[0])

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

def dist_cov_comp(X, Y):
    com_dist = np.sum((X[:2]-Y[:2])**2)
    cov_dist = abs(X[-1]-Y[-1])/max(X[-1], Y[-1])
        
    return (1 + com_dist) * (1 + cov_dist)

def dist_cov(X, Y):
    cov_dist = abs(X[-1]-Y[-1])/max(X[-1], Y[-1])
        
    return cov_dist  

def evaluate_corrected_labels(graph):
    assigned_truths = []
    assigned_labels = []

    for v in graph.vs:
        if v["ground_truth"] == "unclassified":
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

def correct_using_topology(graph):
    # set corrected label and assigned labels as same
    for v in graph.vs:
        if v["assigned_label"] != "unclassified":
            v["corrected_label"] = v["assigned_label"]

    # For each unlabelled vertex, obtain the all the reachable labelled vertices
    contig_topological_neighbours = defaultdict(dict)

    for v in tqdm(graph.vs, total=len(graph.vs), desc="Running BFS from labelled vertices."):
        if v["assigned_label"] != "unclassified":
            if len(graph.neighbors(v)) == 0:
                continue
            
            visited = set()
            queue = [v["id"]]
            level = {}
            
            while queue:
                active = queue.pop(0)
                visited.add(active)
                
                if active not in level:
                    level[active] = 0

                for n in graph.neighbors(active):
                    if n not in visited:
                        visited.add(n)
                        level[n] = level[active] + 1
                        queue.append(n)
            
            for node_id, level in level.items():
                node = graph.vs[node_id]
                if node["assigned_label"] != "unclassified":
                    continue
                contig_topological_neighbours[node_id][v["id"]] = level

    # Correct labels using reachable labelled vertices, coverage and composition
    for v in tqdm(graph.vs, total=len(graph.vs), desc="Label propagation using toplogy+coverage+composition."):
        if v["assigned_label"] == "unclassified" and len(graph.neighbors(v)) > 0 and v["length"] > 1000:
            topological_neighbours = contig_topological_neighbours[v["id"]]
            reachable_labels = set()
            reachables = []
            
            for node_id, distance in topological_neighbours.items():
                reachable_labels.add(graph.vs[node_id]["assigned_label"])
                
                percentage_coverage_distance = abs(graph.vs[node_id]["coverage"]-v["coverage"])/v["coverage"]
                reachables_data = [percentage_coverage_distance, distance, graph.vs[node_id]["assigned_label"]]
                
                if graph.vs[node_id]["length"] > 1000:
                    composition_distance = np.sum((graph.vs[node_id]["profile"]-v["profile"])**2)
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
                
                percentage_coverage_distance = abs(graph.vs[node_id]["coverage"]-v["coverage"])/v["coverage"]
                reachables_data = [percentage_coverage_distance, distance, graph.vs[node_id]["assigned_label"]]
                    
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
    all_missed_coverages = np.array([v["coverage"] for v in graph.vs if v["corrected_label"] =="unclassified"]).reshape(-1, 1)

    # test to check if there are unclassified vertices
    if len(all_missed_coverages) > 0:
        logger.debug(f"Unclassified count = {len(all_missed_coverages)}")
        classified_coverages = [v["coverage"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] > 1000 and len(graph.neighbors(v)) < 3]
        classified_labels = [v["corrected_label"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] > 1000 and len(graph.neighbors(v)) < 3]

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

        if len(neighbor_ids) == 1:
            u["corrected_label"] = graph.vs[neighbor_ids[0]]["corrected_label"]
    
    logger.debug("Refining labels completed.")

    return graph