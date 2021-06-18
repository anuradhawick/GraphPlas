from collections import defaultdict
import logging
import math
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from scipy.stats import poisson
from graphplas_utils import utils, label_prop
import gc
import operator

logger = logging.getLogger('GraphPlas')

mu_intra, sigma_intra = 0, 0.01037897 / 2.
mu_inter, sigma_inter = 0.0676654, 0.03419337

class KNN_custom():
    
    def __init__(self, metric, K=5, n_jobs=8):
        self.K = K
        self.metric=metric
        self.n_jobs=n_jobs

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
        
    def _predict_one(self, X_test_i):
        dist = np.array([self.metric(X_test_i, x_t) for x_t in self.X_train if self.metric(X_test_i, x_t) != float('inf')])
        y_relevant = np.array([y_t for x_t, y_t in zip(self.X_train, self.Y_train) if self.metric(X_test_i, x_t) != float('inf')])
        dist_sorted = dist.argsort()[:self.K]
        
        if len(dist) == 0:
            return "unclassified"
        else:
            neigh_count = {}
            for idx in dist_sorted:
                if y_relevant[idx] in neigh_count:
                    neigh_count[y_relevant[idx]] += 1
                else:
                    neigh_count[y_relevant[idx]] = 1

            sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)

            return sorted_neigh_count[0][0]
        
    def predict(self, X_test):        
        predictions = []
        pool = Pool(self.n_jobs)
        predictions = pool.map(self._predict_one, X_test)
        
        return predictions

def obtain_prob_thresholds(probs, classifier):
    probs_sorted = sorted(probs, reverse=True)
    above_thresh = sum([1 for x in probs_sorted if x >= 0.5])
    below_thresh = len(probs_sorted) - above_thresh

    if classifier == 'pf':
        plasmids = [x for x in probs_sorted if x >= 0.5][0: round(0.2 * above_thresh)]
    else:
        plasmids = [x for x in probs_sorted if x >= 0.5][0: round(0.1 * above_thresh)]

    chromosomes = [x for x in probs_sorted[::-1] if x < 0.5][0: round(0.5 * below_thresh)]

    return plasmids[-1], chromosomes[-1]

def distance_func(data):
    if len(data)==4:
        return data[2] + data[0] + data[1]
    else:
        return data[1] + data[0]

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

def dist_com_GT(vec_1, vec_2):
    eu_distance = np.sum((vec_1 - vec_2)**2)**0.5
    prob_comp = normpdf(eu_distance, mu_intra, sigma_intra) / (normpdf(eu_distance, mu_intra, sigma_intra) + normpdf(eu_distance, mu_inter, sigma_inter))

    if prob_comp == 0:
        return float('inf')
    
    com_dist = -np.log10(prob_comp)
        
    return com_dist

def dist_cov_GT(cov_1, cov_2):
    prob_cov = poisson.pmf(int(cov_1), int(cov_2))
    
    if prob_cov == 0:
        return float('inf')
    
    cov_dist = -np.log10(prob_cov)
    
    return cov_dist

def dist_com_KNN(vec_1, vec_2):
    eu_distance = np.sum((vec_1 - vec_2)**2)**0.5
    prob_comp = normpdf(eu_distance, mu_intra, sigma_intra) / (normpdf(eu_distance, mu_intra, sigma_intra) + normpdf(eu_distance, mu_inter, sigma_inter))
    
    if prob_comp == 0:
        return float('inf')
    
    com_dist = -np.log10(prob_comp)
        
    return com_dist

def dist_cov_KNN(X, Y):
    prob_cov = poisson.pmf(int(X), int(Y))
    
    if prob_cov == 0:
        return float('inf')
    
    cov_dist = -np.log10(prob_cov)
    
    return cov_dist

def dist_cov_comp_KNN(X, Y):
    X_com = X[:-1]
    Y_com = Y[:-1]
    X_cov = X[-1]
    Y_cov = Y[-1]
        
    com_dist = dist_com_KNN(X_com, Y_com)
    cov_dist = dist_cov_KNN(X_cov, Y_cov)
                        
    return com_dist + cov_dist

def evaluate_corrected_labels(graph):
    assigned_truths = []
    assigned_labels = []

    for v in graph.vs:
        if v["ground_truth"] == "unclassified" or v["length"] < 500:
            continue

        assigned_truths.append(v["ground_truth"])
        assigned_labels.append(v["corrected_label"])

    return f"Evaluation results {utils.evaluate(assigned_truths, assigned_labels)}"

# def scale_freqs(contig_profile):
#     logger.debug(f"Scaling the contig trimer profiles. No. of rofiles = {len(contig_profile)}")
#     keys = [key for key in list(contig_profile.keys())]
#     profiles = MinMaxScaler().fit_transform(np.array([contig_profile[key] for key in keys]))
#     contig_profile = {k: p for k, p in zip(keys, profiles)}
#     logger.debug(f"Scaling the contig trimer profiles completed.")

#     return contig_profile


def label_prop_mat(transition_matrix, walk_probabilities, diff, max_iter, labelled):
    itr = 0
    current_diff = np.inf

    # float32 improves performance
    T = transition_matrix.astype('float32')
    Y = walk_probabilities.astype('float32')
    Y_init = np.copy(Y).astype('float32')
    
    Y1 = Y

    labelled_mask = [False for itr in range(Y_init.shape[0])]
    for l in labelled:
        labelled_mask[l] = True
    
    place_holder_1 = np.zeros((T.shape[0], Y.shape[1]))
    Y1 = np.zeros((T.shape[0], Y.shape[1]))
    Y0 = np.zeros((T.shape[0], Y.shape[1]))
    np.copyto(Y1, Y) 

    while current_diff > diff and itr < max_iter:   
        # Y0 = Y1
        np.copyto(Y0, Y1)
        place_holder_1[:] = 0
        # Y1 = T x Y0
        np.matmul(T, Y0, place_holder_1)
        np.copyto(Y1, place_holder_1)

        for i in range(Y_init.shape[0]):
            if labelled_mask[i]:
                Y1[i][i] = Y_init[i][i]
            else:
                row_sum = np.sum(Y1[i]) 
                if row_sum > 0:
                    Y1[i] /= row_sum
        
        # Get difference between values of Y(t+1) and Y(t)
        current_diff = np.sum(np.abs(Y1-Y0))
        itr += 1

        logger.debug(f"Iteration {itr}, diff {current_diff}")     
                
    return Y1

def correct_using_topology(graph):
    # set corrected label and assigned labels as same
    for v in graph.vs:
        if v["assigned_label"] != "unclassified":
            v["corrected_label"] = v["assigned_label"]

    use_new = False

    # === start transform data to fit in LabelProp algorithm ===
    if use_new:
        data = []

        for v in graph.vs:
            vid = v["id"]+1
            degree = len(graph.neighbors(v))

            if degree == 0:
                continue
            elif v["assigned_label"] != "unclassified":
                vlabel = 0            
            else:
                # we treat each contig as belonging to a different class
                vlabel = vid

            neighbour_weights = [[graph.vs[ni]['id']+1, 1.0] for ni in graph.neighbors(v)]
            
            data.append([vid, vlabel, neighbour_weights])

        lp = label_prop.LabelProp()
        lp.load_data_from_mem(data)

        logger.debug(f"Starting label propagation algorithm: iterations {1000}, threshold {0.00001}")
        ans = lp.run(0.00001, 1000, show_log=True, clean_result=False)
        logger.debug(f"Starting label propagation algorithm completed")
        
        del data
        gc.collect()

        # had to use this as vid 0 will be treated as unlabelled 
        vid_map = {v["id"]+1:v["id"] for v in graph.vs}

        # dictionary key: node_id from graph
        #            value: dictionary key: node_id from graph
        #                              value: distance (inverse probability)
        contig_topological_neighbours = {}

        for resline in ans:
            contig_topological_neighbours[vid_map[resline[0]]] = {}

            for [label, probability] in resline[2:]:
                if probability > 0:
                    contig_topological_neighbours[vid_map[resline[0]]][vid_map[label]] = 1 / probability
        del vid_map
        gc.collect()
    # === end transform data to fit in LabelProp algorithm === 

    # === start old implementation
    else:
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

        logger.debug(f"Degree matrix size {degree_matrix.shape}")
        logger.debug(f"Adjacency matrix size{adjacency_matrix.shape}")
        logger.debug(f"Walk probabilities matrix size {walk_probabilities.shape}")

        ## D-1 * A
        logger.debug(f"Inverting degree matrix")
        # degree_matrix_inv = np.linalg.inv(degree_matrix)
        degree_matrix_inv = np.zeros_like(degree_matrix)

        for i in range(len(degree_matrix)):
            if degree_matrix[i][i] != 0:
                degree_matrix_inv[i][i] = 1.0/float(degree_matrix[i][i])

        logger.debug(f"Inverting degree matrix complete")

        transition_matrix = np.matmul(degree_matrix_inv, adjacency_matrix)

        logger.debug(f"Starting label propagation algorithm: iterations {1000}, threshold {0.00001}")
        contig_label_probabilities = label_prop_mat(transition_matrix, walk_probabilities,  0.00001, 1000, {i for i in range(len(labelled_vertices))})
        logger.debug(f"Finished label propagation algorithm")
        contig_topological_neighbours = { v["id"]: { labelled_vertices[i]["id"]: 1/prob for i, prob in enumerate(contig_label_probabilities[label_index_map[v["label_raw"]]]) if prob > 0 } for  v in unlabelled_vertices }
    # === end old implementation

    # Correct labels using reachable labelled vertices, coverage and composition
    for v in graph.vs:
        if v["assigned_label"] == "unclassified" and len(graph.neighbors(v)) > 0 and v["length"] >= 1000:
            topological_neighbours = contig_topological_neighbours[v["id"]]
            reachable_labels = set()
            reachables = []
            
            # for each node, dist from label prop
            # get the set of reachable node ids
            # calculate distance metric (from paper)
            # if there are multiple reachables use voting else its all good

            for node_id, distance in topological_neighbours.items():
                reachable_labels.add(graph.vs[node_id]["assigned_label"])
                
                coverage_distance = dist_cov_GT(graph.vs[node_id]["coverage"], v["coverage"])
                reachables_data = [coverage_distance, distance, graph.vs[node_id]["assigned_label"]]
                
                if graph.vs[node_id]["length"] >= 1000:        
                    composition_distance = dist_com_GT(graph.vs[node_id]["profile"], v["profile"])
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
                
                coverage_distance = dist_cov_GT(graph.vs[node_id]["coverage"], v["coverage"])
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
    to_predict_profiles = np.array([list(v["profile"]) + [v["coverage"]] for v in graph.vs if v["corrected_label"]=="unclassified"  and v["length"] >= 1000])

    # test to check if there are unclassified vertices
    if len(to_predict_profiles) > 0:
        logger.debug(f"Unclassified count = {len(to_predict_profiles)}")
        classified_profiles = np.array([list(v["profile"]) + [v["coverage"]] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] >= 1000 and len(graph.neighbors(v)) < 3])
        classified_labels = [v["corrected_label"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] >= 1000 and len(graph.neighbors(v)) < 3]

        # test to check if there are classified vertices meeting the criteria
        if len(classified_labels) > 0:
            logger.debug(f"Classified count = {len(classified_labels)}")
            if len(classified_labels) * len(to_predict_profiles) > 100000:
                # in order to remove over weighting of coverage
                classified_profiles = MinMaxScaler().fit_transform(classified_profiles)
                neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=threads)
                neigh.fit(classified_profiles, classified_labels)
            else:
                neigh = KNN_custom(metric=dist_cov_comp_KNN, n_jobs=threads)
                neigh.fit(classified_profiles, classified_labels)

            predictions = neigh.predict(to_predict_profiles)   

            for v, p in zip([v for v in graph.vs if v["corrected_label"]=="unclassified"  and v["length"] >= 1000], predictions):
                v["corrected_label"] = p

    logger.debug("Correction using coverage and composition completed.")

    return graph

def correct_using_cov(graph, min_contig_length, threads):
    logger.debug("Correction using coverage.")
    all_missed_coverages = np.array([v["coverage"] for v in graph.vs if v["corrected_label"] =="unclassified" and v["length"] > min_contig_length]).reshape(-1, 1)

    # test to check if there are unclassified vertices
    if len(all_missed_coverages) > 0:
        logger.debug(f"Unclassified count = {len(all_missed_coverages)}")
        classified_coverages = [v["coverage"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] >= 1000 and len(graph.neighbors(v)) < 3]
        classified_labels = [v["corrected_label"] for v in graph.vs if v["corrected_label"]!="unclassified" and v["length"] >= 1000 and len(graph.neighbors(v)) < 3]

        # test to check if there are classified vertices meeting the criteria
        if len(classified_labels) > 0:
            logger.debug(f"Classified count = {len(classified_labels)}")
            classified_coverages = np.array(classified_coverages).reshape(-1, 1)

            if len(classified_labels) * len(all_missed_coverages) > 100000:
                logger.debug(f"Too many iterations to use Poisson PMF, falling back to classical KNN")
                neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=threads)
                neigh.fit(classified_coverages, classified_labels)
            else:
                neigh = KNN_custom(metric=dist_cov_KNN, n_jobs=threads)
                neigh.fit(classified_coverages, classified_labels)

            predictions = neigh.predict(all_missed_coverages)   

            for v, p in zip([v for v in graph.vs if v["corrected_label"] =="unclassified" and v["length"] > min_contig_length], predictions):
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