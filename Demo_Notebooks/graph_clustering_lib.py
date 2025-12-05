"""
Library for graph clustering. 
Includes:
- Building similarity matrix from couples DataFrame
- Generating random couples DataFrame
- Graph visualization utility
- Louvain clustering with resolution scan
- Girvan-Newman clustering with modularity selection
- Graph comparison metrics (Adjusted Rand Index, Jaccard similarity)
- Exporting clusters to HDF5

Use: graph_clustering.ipynb
"""
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Tuple, List, Dict, Any

import networkx as nx
from networkx.algorithms.community import louvain_communities, girvan_newman, greedy_modularity_communities, modularity
from networkx.algorithms.community import modularity as nx_modularity

from sklearn.metrics import adjusted_rand_score
from itertools import combinations

# ---------------------------------------------
#    horoscope-extracted graph
# ---------------------------------------------
def make_simmatrix_from_couples(
    df: pd.DataFrame,
    threshold: float = 1.0,
    id1_col: str = "ID1",
    id2_col: str = "ID2",
    sim_col: str = "similarity",
    scaling: bool = False,
    include_full_range: bool = True,
    dtype: Any = float
) -> Tuple[sparse.csr_matrix, List[Any], Dict[Any, int]]:
    """
    Build a symmetric similarity matrix from a dataframe of couples.
    (IDs preserved so we can relabel graph nodes with original IDs.)
    """
    if id1_col not in df.columns or id2_col not in df.columns or sim_col not in df.columns:
        raise ValueError(f"DataFrame must contain {id1_col}, {id2_col}, {sim_col}")

    if scaling:
        min_sim = df[sim_col].min(); max_sim = df[sim_col].max()
        if max_sim > min_sim:
            df[sim_col] = (df[sim_col] - min_sim) / (max_sim - min_sim)
        else:
            df[sim_col] = 1.0

    # Keep only edges >= threshold
    df = df[df[sim_col] >= threshold].copy()

    # Canonicalize pairs to undirected (min, max) using a stable string-based key,
    # then aggregate duplicates by maximum similarity to avoid double counting.
    def _canon_pair(a, b):
        pa = sorted([a, b], key=lambda x: str(x))
        return pa[0], pa[1]

    canon = df[[id1_col, id2_col]].apply(lambda r: _canon_pair(r[id1_col], r[id2_col]), axis=1)
    df['u'] = [t[0] for t in canon]
    df['v'] = [t[1] for t in canon]
    df_agg = df.groupby(['u', 'v'], as_index=False)[sim_col].max()

    # Build ID universe from aggregated pairs only
    ids = pd.Index(df_agg['u'].dropna()).append(pd.Index(df_agg['v'].dropna())).unique()
    integer_like = False
    try:
        casted = np.array([int(x) for x in ids]); integer_like = True
    except Exception:
        integer_like = False

    if include_full_range and integer_like:
        min_id = int(casted.min()); max_id = int(casted.max())
        id_list = list(range(min_id, max_id + 1))
    else:
        try:
            numeric_vals = np.array([float(x) for x in ids])
            order = np.argsort(numeric_vals)
            id_list = list(ids[order])
        except Exception:
            id_list = sorted(ids, key=lambda x: str(x))

    id_to_index = {id_val: idx for idx, id_val in enumerate(id_list)}
    n = len(id_list)
    rows, cols, data = [], [], []
    for _, row in df_agg.iterrows():
        a, b, val = row['u'], row['v'], row[sim_col]
        if pd.isna(a) or pd.isna(b) or pd.isna(val):
            continue
        if a not in id_to_index or b not in id_to_index:
            continue
        i, j = id_to_index[a], id_to_index[b]
        if i == j:
            # skip self-loops
            continue
        # Add one direction only; we'll symmetrize without summation
        rows.append(i); cols.append(j); data.append(val)
    coo = sparse.coo_matrix((np.array(data, dtype=dtype), (np.array(rows), np.array(cols))), shape=(n, n))
    coo.sum_duplicates()
    # Symmetrize by taking maximum (not sum) to avoid doubling weights
    S_upper = coo.tocsr()
    S = S_upper.maximum(S_upper.T)
    return S, id_list, id_to_index


# ---------------------------------------------
#                random graph
# ---------------------------------------------
def generate_random_couples_df(
    n_ids: int,
    n_pairs: int,
    id_start: int = 0,
    sim_low: float = 0.0,
    sim_high: float = 1.0,
    seed: int = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.arange(id_start, id_start + n_ids)
    ID1 = rng.choice(ids, size=n_pairs, replace=True)
    ID2 = rng.choice(ids, size=n_pairs, replace=True)
    mask = ID1 != ID2
    ID1 = ID1[mask]; ID2 = ID2[mask]
    n_final = len(ID1)
    sim = rng.uniform(sim_low, sim_high, size=n_final)
    return pd.DataFrame({"ID1": ID1, "ID2": ID2, "similarity": sim})

def gen_matrix(N, max_val=10, lam=2.0, seed=42):
    if N <= 0: raise ValueError("N must be positive.")
    rng = np.random.default_rng(seed)
    vals = rng.poisson(lam=lam, size=(N, N))
    vals = np.clip(vals, 0, max_val).astype(int)
    upper = np.triu(vals)
    mat = upper + upper.T - np.diag(np.diag(upper))
    return mat


# ---------------------------------------------
#    graph visualization and export utility
# ---------------------------------------------
# Simple visualization of the best partition (updated sizing/spacing)
def visualize_graph(G, labels=None, title=None, hide_isolated=True,
                     node_size=180, spacing_factor=2.0, edge_alpha=0.9,
                     nb_clusters=None):
    """Visualize graph with nodes colored by cluster labels and edges by weight.

    Improvements:
    - Smaller node circles via `node_size` (default 180).
    - Increased spacing between nodes using spring_layout `k = spacing_factor / sqrt(n)`.
    - Slight edge emphasis (alpha) so edges are more visible.

    Parameters
    ----------
    G : networkx.Graph
    labels : sequence or None
        Cluster labels aligned with sorted(G.nodes()).
    title : str or None
    hide_isolated : bool
        If True, drop degree-0 nodes (count annotated).
    node_size : int
        Size of nodes passed to draw_networkx_nodes.
    spacing_factor : float
        Multiplier controlling layout spread; larger -> more space.
    edge_alpha : float
        Alpha transparency for edges.

    Additional Parameters
    ---------------------
    nb_clusters : int or None
        If provided and labels are given, only plot nodes belonging to the
        largest `nb_clusters` clusters (by size) determined from `labels`.

    Returns
    -------
    dict with keys:
        'isolated_count'
        'shown_nodes'
        'layout_k' (the k value used for spring_layout)
    """
    node_order = sorted(G.nodes())  # global ordering reference
    idx_map = {node:i for i,node in enumerate(node_order)}

    # If requested, limit to the largest nb_clusters by label size
    selected_nodes = set(G.nodes())
    if nb_clusters is not None and labels is not None:
        # compute cluster sizes
        labels_arr = np.asarray(labels)
        unique, counts = np.unique(labels_arr, return_counts=True)
        order = np.argsort(-counts)  # descending by size
        top_labels = set(unique[order][:int(nb_clusters)])
        selected_nodes = {n for n in G.nodes() if labels_arr[idx_map[n]] in top_labels}

    # Build subgraph of selected nodes first
    H0 = G.subgraph(selected_nodes).copy()

    # Optionally hide isolated nodes (after cluster filtering)
    isolated_nodes = [n for n, d in H0.degree() if d == 0]
    nodes_to_plot = [n for n in H0.nodes() if n not in isolated_nodes] if hide_isolated else list(H0.nodes())
    H = H0.subgraph(nodes_to_plot).copy()
    if H.number_of_nodes() == 0:
        print(f"No nodes to plot (after filtering). Isolated count: {len(isolated_nodes)}")
        return {'isolated_count': len(isolated_nodes), 'shown_nodes': [], 'layout_k': None}
    if labels is None:
        label_values = [0 for _ in H.nodes()]
    else:
        label_values = [labels[idx_map[n]] for n in H.nodes()]
    # Compute k for spring layout to expand spacing
    layout_k = spacing_factor / (H.number_of_nodes() ** 0.5)
    pos = nx.spring_layout(H, seed=42, k=layout_k)
    node_cmap = plt.get_cmap('tab20')
    edge_cmap = plt.get_cmap('viridis')
    weights = [d.get('weight', 1) for *_, d in H.edges(data=True)]
    w_min, w_max = (min(weights), max(weights)) if len(weights) else (0, 1)
    norm = Normalize(vmin=w_min, vmax=w_max if w_max > w_min else w_min + 1)
    edge_widths = [1.0 + 2.5 * norm(w) for w in weights] if len(weights) else 1.0
    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=label_values, cmap=node_cmap, node_size=node_size, linewidths=0.8, edgecolors='black')
    if len(weights):
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color=weights, edge_cmap=edge_cmap, edge_vmin=w_min, edge_vmax=w_max, width=edge_widths, alpha=edge_alpha)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=7)
    if len(weights):
        sm = plt.cm.ScalarMappable(norm=norm, cmap=edge_cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.65, label='edge weight')
    if hide_isolated:
        ax.text(0.02, 0.02, f"isolated: {len(isolated_nodes)}", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    if title: ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout(); plt.show()
    return {'isolated_count': len(isolated_nodes), 'shown_nodes': nodes_to_plot, 'layout_k': layout_k}

def export_clusters(df, labels, node_ids, method_name, subset_name="horoscope_full", folder="..\\pre_results\\clusters\\", threshold=None):
    """
    Export clusters into a single HDF5 file for the given method.

    Params:
    - df: Original DataFrame containing texts and metadata
    - labels: List/Array of cluster labels (in the order of node_ids)
    - node_ids: List of IDs corresponding to the labels (sorted(G.nodes()))
    - method_name: 'louvain' or 'girvan_newman'
    - subset_name: Name of the subset (e.g., 'horoscope_full')
    """

    # 1. Create the mapping DataFrame (ID -> Cluster)
    cluster_mapping = pd.DataFrame({
        'ID': node_ids,
        'cluster_label': labels
    })

    # 2. Merge with the original data
    # Use 'inner' to keep only nodes that were in the graph
    # (those filtered by the threshold are excluded)
    result_df = pd.merge(df, cluster_mapping, on='ID', how='inner')

    # 3. Organize columns and sort
    # Ensure the cluster column appears first for readability
    cols = ['cluster_label'] + [c for c in result_df.columns if c != 'cluster_label']
    result_df = result_df[cols].sort_values(by=['cluster_label', 'ID'])

    # 4. Save to HDF5
    filename = f"{folder}{subset_name}_{method_name}.h5"

    # format='table' and data_columns=True enable SQL-like queries later
    # e.g., pd.read_hdf(..., where='cluster_label == 5')
    result_df.to_hdf(filename, key='clusters', mode='w', format='table', data_columns=['cluster_label', 'sign', 'category'])

    print(f" Saved : {filename}")
    print(f"   - Number of elements : {len(result_df)}")
    print(f"   - Number of clusters : {result_df['cluster_label'].nunique()}")

    # Small preview
    return result_df.head()

def least_connected_pairs_from_h5(h5_path: str, G: nx.Graph, output_csv_path: str,
                                  key: str = 'clusters', weight_attr: str = 'weight') -> None:
    """Read a clustering HDF5 file and write, for each cluster of size >= 3,
    the least-connected pair of horoscopes (IDs) and their similarity measure.

    Similarity measure is defined as the bottleneck connectivity between the two nodes:
    - If nodes are connected, it is the minimum edge weight along the path between them
      (which equals the minimum edge on the path in the maximum spanning tree).
    - If there is no path between them within the cluster subgraph, it is 0.

    Outputs a CSV with columns: 'Edge Weight', 'ID1', 'ID2', 'Cluster Label'.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file produced by export_clusters.
    G : networkx.Graph
        The graph containing edge weights.
    output_csv_path : str
        Destination path for the output CSV file.
    key : str
        HDF5 key used when saving (default 'clusters').
    weight_attr : str
        Edge attribute name that stores weight (default 'weight').
    """
    # Load clustering result
    df_clusters = pd.read_hdf(h5_path, key=key)
    # Ensure expected columns
    if 'ID' not in df_clusters.columns or 'cluster_label' not in df_clusters.columns:
        raise ValueError("HDF5 must contain 'ID' and 'cluster_label' columns")

    # Group by cluster_label
    grouped = df_clusters.groupby('cluster_label')['ID'].apply(list)

    rows = []
    for clabel, ids in grouped.items():
        if len(ids) < 3:
            continue  # ignore singletons and couples
        # Build subgraph induced by ids
        Hc = G.subgraph(ids).copy()
        # If disconnected: choose any pair from different components, weight=0
        comps = list(nx.connected_components(Hc.to_undirected() if Hc.is_directed() else Hc))
        if len(comps) > 1:
            # Pick first two components and first nodes from each
            a = next(iter(comps[0]))
            b = next(iter(comps[1]))
            rows.append({'ID1': a, 'ID2': b, 'Cluster Label': clabel, 'Edge Weight': 0.0})
            continue

        # Connected: use maximum spanning tree to get bottleneck
        Hc_u = Hc.to_undirected() if Hc.is_directed() else Hc
        # NetworkX maximum_spanning_tree uses 'weight' attr by default; map if different
        if weight_attr != 'weight':
            # Relabel edge attribute to 'weight' on a copy
            Htmp = nx.Graph()
            Htmp.add_nodes_from(Hc_u.nodes(data=True))
            for u, v, d in Hc_u.edges(data=True):
                w = float(d.get(weight_attr, 0.0))
                Htmp.add_edge(u, v, weight=w)
            Hc_u = Htmp
        MST = nx.maximum_spanning_tree(Hc_u, weight='weight')
        # Find the global minimum edge in the MST
        min_w = None
        min_edge = None
        for u, v, d in MST.edges(data=True):
            w = float(d.get('weight', 0.0))
            if (min_w is None) or (w < min_w):
                min_w = w
                min_edge = (u, v)
        if min_edge is not None:
            rows.append({'ID1': min_edge[0], 'ID2': min_edge[1], 'Cluster Label': clabel, 'similarity estimation': float(min_w)})

    # Write CSV
    out_df = pd.DataFrame(rows, columns=['ID1', 'ID2', 'Cluster Label', 'similarity estimation'])
    out_df.to_csv(output_csv_path, index=False)
    print(f"Written least-connected pairs CSV to: {output_csv_path}")

# ---------------------------------------------
#           Spot Existing Clusters
# ---------------------------------------------
def pre_analyze_graph(G: nx.Graph) -> Dict[str, Any]:
    """Pre-analyze the graph before clustering.

    - Detect connected components (treat directed graphs as undirected for components).
    - Create a label array where each node's label is its component id.
    - Export these component labels using export_clusters.
    - Visualize only the largest components using visualize_graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph to analyze.

    Returns
    -------
    dict with keys:
        'labels' : np.ndarray of component labels aligned to sorted(G.nodes())
        'component_sizes' : list of (label, size) sorted descending by size
        'n_components' : int
        'n_isolated' : int
    """
    H = G.to_undirected() if G.is_directed() else G
    node_order = sorted(H.nodes())
    idx_map = {n: i for i, n in enumerate(node_order)}

    comps = list(nx.connected_components(H))
    comps_sorted = sorted(comps, key=lambda s: len(s), reverse=True)
    # Map each node to a component id
    comp_id_by_node = {}
    for cid, comp in enumerate(comps_sorted):
        for n in comp:
            comp_id_by_node[n] = cid
    labels = np.array([comp_id_by_node.get(n, -1) for n in node_order], dtype=int)

    # Component sizes summary
    sizes = [len(comp) for comp in comps_sorted]

    print(f"Pre-analyze: found {len(comps_sorted)} connected components. Largest sizes: {sizes[:5]}")

    # Modularity of the components partition
    if H.number_of_edges() == 0:
        Q_components = 0.0
    else:
        Q_components = modularity(H, tuple(comps_sorted), weight='weight')

    return {
        'labels': labels,
        'component_sizes': sizes,
        'n_components': len(comps_sorted),
        'Q': Q_components,
        'n_isolated': sum(1 for comp in comps_sorted if len(comp) == 1)
    }


# ---------------------------------------------
#              Louvain Clustering
# ---------------------------------------------
def communities_to_labels(communities, node_order):
    """Convert list of sets to label array following provided node_order list."""
    labels = np.empty(len(node_order), dtype=int)
    idx_map = {node:i for i,node in enumerate(node_order)}
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[idx_map[node]] = cid
    return labels

def run_louvain_resolution_scan(G, resolutions=(0.5, 0.8, 1.0, 1.2, 1.5), seeds=(0, 1, 2), weight='weight'):
    results = []
    for r in resolutions:
        for s in seeds:
            comms = louvain_communities(G, weight=weight, resolution=r, seed=s)
            mod = modularity(G, comms, weight=weight)
            results.append({'resolution': r, 'seed': s, 'modularity': mod, 'n_comm': len(comms), 'communities': comms})
    return results

def summarize_results(results):
    summary = {}
    for r in sorted({d['resolution'] for d in results}):
        mods = [d['modularity'] for d in results if d['resolution'] == r]
        n_comms = [d['n_comm'] for d in results if d['resolution'] == r]
        summary[r] = {'mod_mean': np.mean(mods), 'mod_std': np.std(mods), 'n_comm_mean': np.mean(n_comms)}
    return summary

def choose_best_partition(results):
    best = None
    for d in results:
        if best is None or d['modularity'] > best['modularity'] or (d['modularity'] == best['modularity'] and d['n_comm'] < best['n_comm']):
            best = d
    return best

def robustness_ari(results, resolution, G):
    parts = [d for d in results if d['resolution'] == resolution]
    node_order = sorted(G.nodes())
    label_arrays = [communities_to_labels(p['communities'], node_order) for p in parts]
    aris = []
    for (i, lab1), (j, lab2) in combinations(enumerate(label_arrays), 2):
        aris.append(adjusted_rand_score(lab1, lab2))
    return aris


# ---------------------------------------------
#          Girvan-Newman Clustering
# ---------------------------------------------
def run_gn_select_by_modularity(G, max_levels=20, weight='weight'):
    """Run Girvan-Newman and select best partition by modularity up to max_levels.
    
    Parameters
    ----------
    G : networkx.Graph
        max_levels : int
            Maximum number of levels to explore.
        weight : str or None
            Edge attribute for weights. If None, unweighted.
    Returns
    -------
    dict with keys:
        'best_partition' : tuple of sets
        'best_Q' : float
        'history' : list of dicts with keys 'level', 'n_comm', 'Q', 'communities'
    """
    H = G.to_undirected() if G.is_directed() else G
    comp_gen = girvan_newman(H)
    history = []
    best = None
    for level, communities in enumerate(comp_gen, start=1):
        comms = tuple(sorted((set(c) for c in communities), key=lambda s: min(s)))
        Q = nx_modularity(H, comms, weight=weight)
        record = {'level': level, 'n_comm': len(comms), 'Q': Q, 'communities': comms}
        history.append(record)
        if best is None or Q > best['Q'] or (Q == best['Q'] and len(comms) < best['n_comm']):
            best = record
        if level >= max_levels or len(comms) >= H.number_of_nodes():
            break
    return {'best_partition': best['communities'], 'best_Q': best['Q'], 'history': history}


# ---------------------------------------------
#              Greedy Clustering
# ---------------------------------------------
def run_greedy(G: nx.Graph):
    """Run greedy modularity maximization clustering."""
    # Ensure we have an undirected view for modularity computation
    H = G.to_undirected() if G.is_directed() else G

    # Compute communities using greedy modularity
    gm_comms = greedy_modularity_communities(H, weight='weight')

    # Convert to labels aligned with sorted node order
    node_order = sorted(G.nodes())
    gm_labels = communities_to_labels(gm_comms, node_order)

    # Compute modularity of the greedy solution
    Q_gm = modularity(H, gm_comms, weight='weight')
    return {'communities': gm_comms, 'labels': gm_labels, 'Q': Q_gm}

# ---------------------------------------------
#          Graph Comparison Metrics
# ---------------------------------------------
def jaccard_partition(y1, y2):
    """Pair-counting Jaccard similarity between two cluster labelings.
    Matches the definition used in exercise10 (counts co-assigned pairs)."""
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    N = len(y1)
    S = 0  # pairs in same cluster in both
    D = 0  # pairs in different clusters in both
    for i in range(N):
        for j in range(i):
            same1 = (y1[i] == y1[j])
            same2 = (y2[i] == y2[j])
            if same1 and same2:
                S += 1
            if (not same1) and (not same2):
                D += 1
    total_pairs = N * (N - 1) / 2
    if total_pairs == D:
        return 0.0
    return S / (total_pairs - D)
