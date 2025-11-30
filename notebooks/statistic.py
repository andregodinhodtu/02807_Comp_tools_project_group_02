import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.dates as mdates

################################################################################
def analyze_clusters_statistics(hdf_file_path, n_keywords=5, min_cluster_size=10):
    """
    Complete statistical analysis of clusters.
    Includes: Keywords (TF-IDF), Dominant Sign, IDs, and Concatenated Text.
    """
    print(f"Loading file: {hdf_file_path}...")
    try:
        df = pd.read_hdf(hdf_file_path, key='clusters')
    except (FileNotFoundError, KeyError):
        print("Error: File inaccessible or key not found.")
        return

    # Basic cleaning
    df = df.dropna(subset=['horoscope'])
    
    # Filter out small clusters
    cluster_counts = df['cluster_label'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_cluster_size].index
    
    if len(valid_clusters) == 0:
        print(f" WARNING: No clusters found with size >= {min_cluster_size}.")
        return

    df_filtered = df[df['cluster_label'].isin(valid_clusters)]
    print(f" Analyzing {len(valid_clusters)} significant clusters...")

    # --- 1. SEMANTIC ANALYSIS ---
    print(" Calculating keywords & Aggregating text...")
    
    # Group text by cluster (Concatenation happens here)
    # This series acts as our 'Full Text' source
    cluster_texts = df_filtered.groupby('cluster_label')['horoscope'].apply(lambda x: " ".join(x))
    
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=1) 
        tfidf_matrix = tfidf.fit_transform(cluster_texts)
        feature_names = np.array(tfidf.get_feature_names_out())
    except ValueError as e:
        print(f" Skipping TF-IDF: {e}")
        return

    # --- 2. REPORT GENERATION ---
    print(" Generating detailed report...")
    
    results = []
    
    for idx, cluster_id in enumerate(cluster_texts.index):
        cluster_data = df[df['cluster_label'] == cluster_id]
        size = len(cluster_data)
        
        # A. Keywords
        tfidf_scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = tfidf_scores.argsort()[::-1][:n_keywords]
        keywords = ", ".join(feature_names[top_indices])
        
        # B. Dominant Sign
        if 'sign' in cluster_data.columns:
            top_sign_count = cluster_data['sign'].value_counts().head(1)
            if not top_sign_count.empty:
                sign_name = top_sign_count.index[0]
                sign_pct = (top_sign_count.values[0] / size) * 100
                sign_info = f"{sign_name} ({sign_pct:.1f}%)"
            else:
                sign_info = "N/A"
        else:
            sign_info = "N/A"

        # C. Representative Prediction (Medoid)
        texts = cluster_data['horoscope'].tolist()
        if len(texts) == 1:
            rep_text = texts[0]
        else:
            try:
                local_tfidf = TfidfVectorizer(stop_words='english')
                local_mat = local_tfidf.fit_transform(texts)
                centroid = np.asarray(local_mat.mean(axis=0))
                sims = cosine_similarity(local_mat, centroid)
                rep_text = texts[sims.argmax()]
            except ValueError:
                rep_text = texts[0]

        rep_short = rep_text[:100] + "..." 
        
        # D. IDs List
        if 'ID' in cluster_data.columns:
            ids_str = ", ".join(map(str, cluster_data['ID'].tolist()))
        else:
            ids_str = ""

        # E. Full Concatenated Text
        # We retrieve it directly from our groupby object
        full_text = cluster_texts[cluster_id]

        results.append({
            'Cluster ID': cluster_id,
            'Size': size,
            'Keywords': keywords,
            'Dominant Sign': sign_info,
            'IDS': ids_str,
            'Representative Example': rep_short,
            'Full Text': full_text  # <--- New Column
        })

    # Display Results
    df_report = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"ANALYSIS RESULTS (Top 10 Clusters by size)")
    print("="*80)
    
    # For display in Notebook, we hide the very long columns
    cols_to_hide = ['Representative Example']
    cols_to_show = [c for c in df_report.columns if c not in cols_to_hide]
    
    print(df_report[cols_to_show].sort_values('Size', ascending=False).head(10).to_string(index=False))
    
    # Export to CSV (Includreing Full Text and IDs)
    report_name = hdf_file_path.replace('.h5', '_full_report.csv')
    df_report.to_csv(report_name, index=False)
    print(f"\n Report with FULL TEXT saved to: {report_name}")

###################################################################
def plot_cluster_details(hdf_file_path, cluster_id):
    """
    Displays Sign, Category, and Date histograms for a specific cluster.
    """
    # 1. Load data (optimized to load only the targeted cluster)
    try:
        # Use 'where' to load only the requested cluster (very fast)
        df_cluster = pd.read_hdf(hdf_file_path, key='clusters', where=f'cluster_label == {cluster_id}')
    except (FileNotFoundError, KeyError):
        print(f"Error: Impossible to find cluster {cluster_id} in file {hdf_file_path}")
        return
    
    if len(df_cluster) == 0:
        print(f"Cluster {cluster_id} is empty or does not exist.")
        return

    # 2. Setup the figure (3 charts side-by-side)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"Cluster {cluster_id} Analysis (n={len(df_cluster)})", fontsize=16)
    
    # --- Sign Distribution ---
    if 'sign' in df_cluster.columns:
        sign_counts = df_cluster['sign'].value_counts().sort_index()
        sign_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title("Distribution by Sign")
        axes[0].set_ylabel("Number of horoscopes")
        axes[0].tick_params(axis='x', rotation=45)
    else:
        axes[0].text(0.5, 0.5, "No 'sign' data", ha='center')

    # --- Category Distribution ---
    if 'category' in df_cluster.columns:
        cat_counts = df_cluster['category'].value_counts()
        cat_counts.plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
        axes[1].set_title("Distribution by Category")
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, "No 'category' data", ha='center')

    # --- Temporal Distribution (Dates) ---
    if 'date' in df_cluster.columns:
        # Convert to datetime if not already done (assuming YYYYMMDD format)
        try:
            dates = pd.to_datetime(df_cluster['date'], format='%Y%m%d', errors='coerce')
        except:
            dates = pd.to_datetime(df_cluster['date'], errors='coerce')
            
        # Remove invalid dates for plotting
        dates = dates.dropna()
        
        if not dates.empty:
            axes[2].hist(dates, bins=30, color='salmon', edgecolor='black')
            axes[2].set_title("Temporal Distribution")
            # Pretty date formatting on X-axis
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())
            axes[2].tick_params(axis='x', rotation=45)
        else:
             axes[2].text(0.5, 0.5, "Invalid date format", ha='center')
    else:
        axes[2].text(0.5, 0.5, "No 'date' data", ha='center')

    plt.tight_layout()
    plt.show()

#######################################################################
def add_advanced_statistics(hdf_file_path, min_cluster_size=10):
    """
    Enriches the analysis with statistics on text length, vocabulary diversity, 
    and semantic cohesion. Handles empty results gracefully.
    """
    print(f"Loading file: {hdf_file_path}...")
    try:
        df = pd.read_hdf(hdf_file_path, key='clusters')
    except (FileNotFoundError, KeyError):
        print("Error: File inaccessible or key not found.")
        return

    # Filter out small clusters (noise)
    counts = df['cluster_label'].value_counts()
    valid_clusters = counts[counts >= min_cluster_size].index
    
    # --- SAFETY CHECK 1: Check if we have clusters ---
    if len(valid_clusters) == 0:
        print(f"WARNING: No clusters found with size >= {min_cluster_size}.")
        print(f"   Max cluster size in data: {counts.max() if len(counts) > 0 else 0}")
        print("   -> Returning empty DataFrame.")
        return pd.DataFrame() # Return empty to avoid crash
    # -------------------------------------------------

    df = df[df['cluster_label'].isin(valid_clusters)]
    
    stats_list = []
    
    print(f"Calculating advanced statistics for {len(valid_clusters)} clusters...")
    
    for cluster_id in valid_clusters:
        sub_df = df[df['cluster_label'] == cluster_id]
        texts = sub_df['horoscope'].fillna("").tolist()
        
        # 1. Length & Vocabulary
        word_counts = [len(t.split()) for t in texts]
        avg_len = np.mean(word_counts) if word_counts else 0
        
        diversities = [len(set(t.split())) / len(t.split()) if len(t.split()) > 0 else 0 for t in texts]
        avg_diversity = np.mean(diversities) if diversities else 0
        
        # 2. Semantic Cohesion
        if len(texts) > 1:
            try:
                tfidf = TfidfVectorizer(stop_words='english')
                matrix = tfidf.fit_transform(texts)
                
                # Fix: Explicit conversion to array
                centroid = np.asarray(matrix.mean(axis=0))
                
                sims = cosine_similarity(matrix, centroid)
                cohesion = np.mean(sims)
            except ValueError:
                cohesion = 0.0
        else:
            cohesion = 1.0 

        stats_list.append({
            'Cluster': cluster_id,
            'Size': len(texts),
            'Avg Words/Text': round(avg_len, 1),
            'Lexical Diversity': round(avg_diversity, 2),
            'Semantic Cohesion': round(cohesion, 3)
        })
        
    # --- SAFETY CHECK 2: Ensure stats_list is not empty ---
    if not stats_list:
        print("No statistics could be calculated.")
        return pd.DataFrame()
        
    # Create Report DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # Sort only if column exists (which it should now)
    if 'Semantic Cohesion' in stats_df.columns:
        stats_df = stats_df.sort_values('Semantic Cohesion', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 MOST COHERENT CLUSTERS")
    print("="*80)
    display(stats_df.head(10))
    
    return stats_df
