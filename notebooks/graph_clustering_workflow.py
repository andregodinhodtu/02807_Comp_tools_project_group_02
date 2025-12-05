# %%
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

# ---------------------------------------------
full_dataset = "horoscope_full"
df = pd.read_csv(f'..\\data\\{full_dataset}.csv')
df.head()

input_method = 1
# 0 for MinHashing matrix
# 1 for Frequent Itemsets matrix
# 2 for Random graph

subset_idd = "full"

# MinHashing input
input_file_name_mh = f"similarity_{subset_idd}_lsh"  # MinHashing

# Frequent Itemsets input
input_file_name_fi = f"sentence_jaccard_{subset_idd}" # Frequent Itemsets

threshold = 0.4 # edges with weight below this value will be ignored
# careful, for frequent itemsets, similarity depends on the input dataset (for example, if id1 and id2 have s1imilarity 0.4 in full dataset, they may have similarity 0.6 in a smaller subset)
methods = [ # select the methods you want to run by commenting
    'components',
    'louvain',
    # 'girvan_newman', # very long to run, use only for small graphs
    'greedy',
]
plot_graph = False
pre_visualization = False

# Number of largest clusters to plot (when labels available)
nb_clusters_to_plot = 30
# ---------------------------------------------

# Dictionary to store runtimes for each method
method_runtimes = {}

input_method_name = "freqitemsets" if input_method == 1 else ("minhashing" if input_method == 0 else "randomgraph")
res_folder = f"..\\pre_results\\clusters_{input_method_name}\\"

# ---------------------------------------------
#   Make similarity matrix from input data
# ---------------------------------------------

# MinHashing
if input_method == 0:
    df_minhash = pd.read_csv(f'..\\pre_results\\minhashing\\{input_file_name_mh}.csv')
    print(df_minhash.head())
    M_thr, id_list, id_to_index = gcltr.make_simmatrix_from_couples(
        df_minhash,
        threshold=threshold,
        sim_col="Similarity",
    )
    print(M_thr)

# frequent itemsets
if input_method == 1:
    df_minhash = pd.read_csv(f'..\\pre_results\\frequent_itemsets\\{input_file_name_fi}.csv')
    print(df_minhash.head())
    M_thr, id_list, id_to_index = gcltr.make_simmatrix_from_couples(
        df_minhash,
        threshold=threshold,
        sim_col="similarity",
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

# Global node order for consistent labeling/exports
node_order = sorted(G.nodes())

# ---------------------------------------------
#           Pre-Analyze: Components
# ---------------------------------------------
if 'components' in methods:
    print("\n--- Pre-analyze graph (components) ---")
    try:
        pre_res = gcltr.pre_analyze_graph(G=G)
        print(f"Components found: {pre_res['n_components']}")
        # Print modularity of the components partition
        if 'Q' in pre_res:
            print(f"Components Modularity Q: {pre_res['Q']:.4f}")
    except Exception as e:
        print("Pre-analyze failed:", e)

    try: # Export as clusters (component labels)
        gcltr.export_clusters(
            df=df,
            labels=pre_res['labels'],
            node_ids=node_order,
            method_name="components",
            subset_name=subset_idd,
            folder=res_folder,
            threshold=threshold
        )
    except Exception as e:
        print("Components export failed:", e)
    
    if plot_graph:
        try: # Visualize only the largest components
            gcltr.visualize_graph(
                G,
                labels=pre_res['labels'],
                title=f"Top {nb_clusters_to_plot} components by size | method : components | subset : {subset_idd}",
                nb_clusters=nb_clusters_to_plot
                )
        except Exception as e:
            print("Components visualization failed:", e)



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
    print(f"Louvain Modularity Q: {best['modularity']:.4f}")

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

    if plot_graph:
        gcltr.visualize_graph(G, best_labels, title=f"Louvain partition (resolution={best['resolution']})", nb_clusters=nb_clusters_to_plot)

    nodes_ordered = sorted(G.nodes())
    
    # Export Louvain clusters
    print("\n--- Export Louvain ---")
    try:
        gcltr.export_clusters(
            df=df, 
            labels=best_labels, 
            node_ids=nodes_ordered, 
            method_name='louvain', 
            subset_name=subset_idd,
            folder=res_folder,
            threshold=threshold
        )
    except Exception as e:
        print('Louvain export failed:', e)

    # Optional: export least connected pairs to assess the quality of the clustering
    # try:
    #     clustering_path_louvain = f"..\\pre_results\\clusters_{input_method_name}\\{subset_idd}_louvain.h5"
    #     gcltr.least_connected_pairs_from_h5(
    #         h5_path=clustering_path_louvain,
    #         G=G,
    #         output_csv_path=f"{clustering_path_louvain.split('.h5')[0]}_least_connected_pairs.csv"
    #     )
    # except Exception as e:
    #     print('Louvain least connected pairs export failed:', e)

    method_runtimes['louvain'] = time.perf_counter() - t0
    print(f"Louvain runtime: {method_runtimes['louvain']:.2f} s")

# ---------------------------------------------
#            Girvan-Newman Clustering
# ---------------------------------------------
if 'girvan_newman' in methods:
    assert 0==1,"Don't run girvan newman, it's too long"
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

    if plot_graph:
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
            subset_name=subset_idd,
            folder=res_folder
        )
    except Exception as e:
        print('GN export failed:', e)

# ---------------------------------------------
#        Greedy Modularity Clustering
# ---------------------------------------------
# Implement Greedy Modularity on the loaded graph
if 'greedy' in methods:
    t0_g = time.perf_counter()
    greedy_result = gcltr.run_greedy(G=G)
    gm_comms = greedy_result['communities']
    greedy_labels = greedy_result['labels']
    Q_gm = greedy_result['Q']
    print(f"Greedy Modularity Q: {Q_gm:.4f}; communities={len(gm_comms)}")

    # Optional: export clusters
    try:
        gcltr.export_clusters(
            df=df,
            labels=greedy_labels,
            node_ids=node_order,
            method_name='greedy',
            subset_name=subset_idd,
            folder=res_folder,
            threshold=threshold
        )
    except Exception as e:
        print('Greedy export failed:', e)
    # Visualize
    if plot_graph:
        gcltr.visualize_graph(
            G,
            greedy_labels,
            title=f"Greedy Modularity partition (Q={Q_gm:.3f}, k={len(gm_comms)})",
            nb_clusters=nb_clusters_to_plot
            )

    # Optional: export least connected pairs to assess the quality of the clustering
    # try:
    #     clustering_path_greedy = f"..\\pre_results\\clusters_{input_method_name}\\{subset_idd}_greedy.h5"
    #     gcltr.least_connected_pairs_from_h5(
    #         h5_path=clustering_path_greedy,
    #         G=G,
    #         output_txt_path=f"{clustering_path_greedy.split('.h5')[0]}_least_connected_pairs.txt"

    #     )
    # except Exception as e:
    #     print('Greedy least connected pairs export failed:', e)

    method_runtimes['greedy'] = time.perf_counter() - t0_g
    print(f"Greedy Modularity runtime: {method_runtimes['greedy']:.2f} s")


# ---------------------------------------------
#    Compare Louvain and Greedy Modularity
# ---------------------------------------------
compare_louvain_greedy = False

# Compare only if both label arrays exist
have_louvain = 'best_labels' in globals()
have_greedy = 'greedy_labels' in globals()
if have_louvain and have_greedy and compare_louvain_greedy:
    assert len(best_labels) == len(greedy_labels), "Label arrays must have same length"
    y_louvain = np.asarray(best_labels)
    y_greedy = np.asarray(greedy_labels)
    metrics = {
        'Rand': rand_score(y_louvain, y_greedy),
        'Adjusted Rand (ARI)': adjusted_rand_score(y_louvain, y_greedy),
        'Jaccard (pair-counting)': gcltr.jaccard_partition(y_louvain, y_greedy),
        'NMI': normalized_mutual_info_score(y_louvain, y_greedy)
    }
    df_metrics = pd.DataFrame(metrics, index=['Louvain vs Greedy']).T
    display(df_metrics)
    print(f"Louvain communities: {len(np.unique(y_louvain))}, Greedy communities: {len(np.unique(y_greedy))}")
    # save comparison in csv file
    df_metrics.to_csv(f"..\\pre_results\\{res_folder}\\{subset_idd}_louvain_vs_greedy.csv")
else:
    print("Skipping Louvain vs Greedy comparison:",
          f"Louvain={'available' if have_louvain else 'missing'},",
          f"Greedy={'available' if have_greedy else 'missing'}")

# ---------------------------------------------
#              Runtime Summary
# ---------------------------------------------
print("\n--- Runtime Summary ---")
for m, rt in method_runtimes.items():
    print(f"{m}: {rt:.2f} s")

# %%
