# %%
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import time
# from matplotlib.colors import Normalize
# from typing import Tuple, List, Dict, Any

import networkx as nx
# from networkx.algorithms.community import louvain_communities, modularity

from sklearn.metrics import adjusted_rand_score, rand_score, normalized_mutual_info_score
from IPython.display import display
# from itertools import combinations

import graph_clustering_lib as gcltr

full_dataset = "horoscope_full"
df = pd.read_csv(f'..\\data\\{full_dataset}.csv')
df.head()

input_method = 0
# 0 for MinHashing matrix
# 1 for Frequent Itemsets matrix
# 2 for Random graph
subset_idd = "general"
input_file_name = f"similarity_{subset_idd}_lsh"  # MinHashing
# input_file_name = "shared_2_itemsets_pairs" # Frequent Itemsets
# !! be sure to set the correct input_file_name according to input_method !!

threshold = 0.3 # edges with weight below this value will be ignored
methods = [ # select the methods you want to run by commenting
    'louvain',
    'girvan_newman',
    'greedy_modularity',
]
pre_visualization = False

# Dictionary to store runtimes for each method
method_runtimes = {}

# ---------------------------------------------
#   Make similarity matrix from input data
# ---------------------------------------------

# MinHashing
if input_method == 0:
    df_minhash = pd.read_csv(f'..\\pre_results\\{input_file_name}.csv')
    print(df_minhash.head())
    M_thr, id_list, id_to_index = gcltr.make_simmatrix_from_couples(
        df_minhash,
        threshold=threshold,
        sim_col="Similarity",
    )
    print(M_thr)

# frequent itemsets
if input_method == 1:
    df_minhash = pd.read_csv(f'..\\pre_results\\{input_file_name}.csv')
    print(df_minhash.head())
    M_thr, id_list, id_to_index = gcltr.make_simmatrix_from_couples(
        df_minhash,
        threshold=threshold,
        sim_col="value",
        scaling=True,
    )
    print(M_thr)

# random graph
if input_method == 2:
    M = gcltr.gen_matrix(10, max_val=5, lam=1.5, seed=5)
    print("Symmetric:", np.all(M == M.T))
    threshold_rand = 0
    M_thr = M.copy()
    np.fill_diagonal(M_thr, 0)
    M_thr[M_thr <= threshold_rand] = 0
    M_thr = M_thr / M_thr.max()
    id_list = list(range(1, M_thr.shape[0]+1))  # original IDs start at 1
    id_to_index = {id_val: idx for idx, id_val in enumerate(id_list)}
    print("After thresholding:\n", M_thr)

# ---------------------------------------------
#    Build graphs (preserve original IDs)
# ---------------------------------------------
A = sparse.csr_array(M_thr)
G_raw = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
node_mapping = {i: id_list[i] for i in range(len(id_list))}
G = nx.relabel_nodes(G_raw, node_mapping)

Gd_raw = nx.from_numpy_array(M_thr, create_using=nx.DiGraph)
Gd_raw.remove_edges_from(nx.selfloop_edges(Gd_raw))
Gd_raw.remove_edges_from([(u, v) for u, v, d in Gd_raw.edges(data=True) if d.get("weight", 0) <= 0])
Gd = nx.relabel_nodes(Gd_raw, node_mapping)

G.number_of_nodes(), G.number_of_edges()

if pre_visualization:
    gcltr.visualize_graph(G)

# ---------------------------------------------
#             Louvain Clustering
# ---------------------------------------------
plot_perf = 0
if 'louvain' in methods:
    t0 = time.perf_counter()
    # Use actual node ordering with original IDs
    node_order = sorted(G.nodes())
    resolutions = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 10.0)
    seeds = range(5)
    results = gcltr.run_louvain_resolution_scan(G, resolutions=resolutions, seeds=seeds)
    summary = gcltr.summarize_results(results)
    best = gcltr.choose_best_partition(results)
    best_labels = gcltr.communities_to_labels(best['communities'], node_order)
    best_aris = gcltr.robustness_ari(results, best['resolution'], G)
    print(f"Best resolution: {best['resolution']}, modularity={best['modularity']:.4f}, communities={best['n_comm']}")
    print(f"Robustness (ARI mean ± std at best resolution): {np.mean(best_aris):.3f} ± {np.std(best_aris):.3f}")

    print("\nResolution  | Modularity(mean±std) | Avg #Communities")
    for r, vals in summary.items():
        print(f"{r:<10} | {vals['mod_mean']:.4f} ± {vals['mod_std']:.4f}    | {vals['n_comm_mean']:.2f}")

    if plot_perf:
        plt.figure(figsize=(6,4))
        plt.errorbar([r for r in summary.keys()], [v['mod_mean'] for v in summary.values()], yerr=[v['mod_std'] for v in summary.values()], fmt='o-', capsize=4)
        plt.xlabel('Resolution'); plt.ylabel('Modularity (mean ± std)'); plt.title('Louvain modularity vs resolution'); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
        plt.figure(figsize=(6,4))
        plt.plot([r for r in summary.keys()], [v['n_comm_mean'] for v in summary.values()], 's--')
        plt.xlabel('Resolution'); plt.ylabel('Avg # Communities'); plt.title('Community count vs resolution'); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    gcltr.visualize_graph(G, best_labels, title=f"Louvain partition (resolution={best['resolution']})")

    nodes_ordered = sorted(G.nodes())
    
    # Export Louvain clusters
    print("\n--- Export Louvain ---")
    try:
        gcltr.export_clusters(
            df=df, 
            labels=best_labels, 
            node_ids=nodes_ordered, 
            method_name='louvain', 
            subset_name=subset_idd
        )
    except Exception as e:
        print('Louvain export failed:', e)

    method_runtimes['louvain'] = time.perf_counter() - t0
    print(f"Louvain runtime: {method_runtimes['louvain']:.2f} s")

# ---------------------------------------------
#            Girvan-Newman Clustering
# ---------------------------------------------
if 'girvan_newman' in methods:
    gn_result = gcltr.run_gn_select_by_modularity(G, max_levels=30, weight='weight')
    node_order = sorted(G.nodes())

    try:
        gn_best = gn_result['best_partition']
        gn_Q = gn_result['best_Q']
        gn_hist = gn_result['history']
        gn_best_labels = gcltr.communities_to_labels(gn_best, node_order)
        print(f"GN best: Q={gn_Q:.4f}, communities={len(gn_best)}")
    except Exception as e:
        print('GN labeling failed:', e)

    if plot_perf:
        x = [h['n_comm'] for h in gn_hist]; y = [h['Q'] for h in gn_hist]
        plt.figure(figsize=(6,4))
        plt.plot(x, y, 'o-')
        plt.xlabel('# Communities'); plt.ylabel('Modularity Q'); plt.title('Girvan–Newman: modularity vs #communities')
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    gcltr.visualize_graph(G, gn_best_labels, title=f"Girvan–Newman partition (Q={gn_Q:.3f}, k={len(gn_best)})")

    nodes_ordered = sorted(G.nodes())
    
    print("\n--- Export Girvan-Newman ---")
    gcltr.export_clusters(
        df=df, 
        labels=gn_best_labels, 
        node_ids=nodes_ordered, 
        method_name='girvan_newman', 
        subset_name=full_dataset
    )
    try:
        gcltr.export_clusters(
            df=df,
            labels=gn_best_labels,
            method_name='girvan_newman',
            subset_name=subset_idd
        )
    except Exception as e:
        print('GN export failed:', e)

# ---------------------------------------------
#        Greedy Modularity Clustering
# ---------------------------------------------
# Implement Greedy Modularity on the loaded graph
if 'greedy_modularity' in methods:
    t0_g = time.perf_counter()
    greedy_result = gcltr.run_greedy(G=G)
    gm_comms = greedy_result['communities']
    greedy_labels = greedy_result['labels']
    Q_gm = greedy_result['Q']
    # Visualize
    gcltr.visualize_graph(G, greedy_labels, title=f"Greedy Modularity partition (Q={Q_gm:.3f}, k={len(gm_comms)})")

    # Optional: export clusters
    try:
        gcltr.export_clusters(
            df=df,
            labels=greedy_labels,
            node_ids=node_order,
            method_name='greedy_modularity',
            subset_name=subset_idd
        )
    except Exception as e:
        print('Greedy export failed:', e)

    method_runtimes['greedy_modularity'] = time.perf_counter() - t0_g
    print(f"Greedy Modularity runtime: {method_runtimes['greedy_modularity']:.2f} s")

# ---------------------------------------------
#    Compare Louvain and Greedy Modularity
# ---------------------------------------------
# Ensure both label arrays exist and align in length
assert 'best_labels' in globals(), "Louvain labels 'best_labels' not found"
assert 'greedy_labels' in globals(), "Greedy Modularity labels 'greedy_labels' not found"
assert len(best_labels) == len(greedy_labels), "Label arrays must have same length"

y_louvain = np.asarray(best_labels)
y_greedy = np.asarray(greedy_labels)

metrics = {
    'Rand': rand_score(y_louvain, y_greedy),
    'Adjusted Rand (ARI)': adjusted_rand_score(y_louvain, y_greedy),
    'Jaccard (pair-counting)': gcltr.jaccard_partition(y_louvain, y_greedy),
    'NMI': normalized_mutual_info_score(y_louvain, y_greedy)
}
df_metrics = pd.DataFrame(metrics, index=['Louvain vs GN']).T
display(df_metrics)

# Optional: quick sanity display of community counts
print(f"Louvain communities: {len(np.unique(y_louvain))}, GN communities: {len(np.unique(y_greedy))}")

# ---------------------------------------------
#              Runtime Summary
# ---------------------------------------------
print("\n--- Runtime Summary ---")
for m, rt in method_runtimes.items():
    print(f"{m}: {rt:.2f} s")