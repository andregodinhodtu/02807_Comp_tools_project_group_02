# ---------------------------------------------------
# MinHashing
# ---------------------------------------------------

full_dataset = "horoscope_full"
df = pd.read_csv(f'..\\data\\{full_dataset}.csv')
df.head()

subset_idd = "career"

# please create the tabular 'df_minhash which is built
#  the same way as the csv files similarity_<subset_idd>_lsh.csv

# ---------------------------------------------------
# Graph Clustering
# ---------------------------------------------------
import pandas as pd
from scipy import sparse
import networkx as nx

# custom graph clustering library
import notebooks.graph_clustering_lib as gcltr

M_thr, id_list, id_to_index = gcltr.make_simmatrix_from_couples(
    df_minhash,
    threshold=0.3,
    sim_col="Similarity",
    )

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
node_order = sorted(G.nodes())

# -----------------------
# Components Clustering
nb_clusters_to_plot = 10
print("\n--- Pre-analyze graph (components) ---")
try:
    pre_info = gcltr.pre_analyze_graph(
        G=G,
        df=df,
        subset_name=subset_idd,
        folder="..\\pre_results\\clusters\\",
        top_k_visualize=nb_clusters_to_plot,
        method_name="components"
    )
    print(f"Components found: {pre_info['n_components']}")
    # Print modularity of the components partition
    if 'Q' in pre_info:
        print(f"Components Modularity Q: {pre_info['Q']:.4f}")
except Exception as e:
    print("Pre-analyze failed:", e)

# ---------------------------------------------------
# Extract Statistics from Clusters
# ---------------------------------------------------
