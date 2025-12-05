import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.dates as mdates
from collections import Counter

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

#########################################################################
def generate_simple_copy_table(base_data_path, reports_folder_path):
    """
    Generates a table of copies assuming that (Copies = Cluster Size - 1).
    """
    
    print("Calculating reference totals...")
    try:
        df_base = pd.read_csv(base_data_path)
        # Count how many 'career', 'love', etc. horoscopes there are in total
        cat_counts = df_base['category'].value_counts().to_dict()
        cat_counts['full'] = len(df_base) # The entire dataset
    except Exception as e:
        print(f"Error loading base data: {e}")
        return

    categories = ['career', 'general', 'love', 'wellness', 'full']
    # Adapt this list to the methods you actually executed
    methods = ['louvain', 'greedy', 'components'] 
    
    results = []

    print("Calculating copies...")
    for cat in categories:
        total_in_category = cat_counts.get(cat, 0)
        
        for method in methods:
            # Filename generated in the previous step
            filename = f"{cat}_{method}_full_report.csv"
            filepath = os.path.join(reports_folder_path, filename)
            
            nb_copies = 0
            
            if os.path.exists(filepath):
                try:
                    # Read only necessary columns for speed
                    df_report = pd.read_csv(filepath, usecols=['Size'])
                    
                    # --- SIMPLIFIED FORMULA ---
                    # Sum of (Size - 1) for all clusters
                    # If size=1, copies=0. If size=10, copies=9.
                    if not df_report.empty:
                        # Ensure no sizes < 1
                        nb_copies = (df_report['Size'] - 1).clip(lower=0).sum()
                        
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
            
            # Calculate rate (%)
            rate = (nb_copies / total_in_category * 100) if total_in_category > 0 else 0

            results.append({
                'Category': cat,
                'Method': method,
                'Total_Items': total_in_category,
                'Copies_Count': nb_copies,
                'Repetition_Rate': round(rate, 2)
            })

    df_res = pd.DataFrame(results)

    # Table 1: Absolute number of copies
    pivot_count = df_res.pivot(index='Category', columns='Method', values='Copies_Count')
    
    # Table 2: Repetition rate in %
    pivot_rate = df_res.pivot(index='Category', columns='Method', values='Repetition_Rate')

    print("\n" + "="*60)
    print("NUMBER OF COPIES (Based on cluster size)")
    print("="*60)
    display(pivot_count)
    
    print("\n" + "="*60)
    print("REPETITION RATE (%)")
    print("="*60)
    display(pivot_rate)
    
    return pivot_count, pivot_rate

###############################################################################
def analyze_components_victimization(base_data_path, reports_folder_path):
    """
    Calculates the recycling rate per sign ONLY for the 'components' method.
    Logic: In a cluster, if a sign appears N times, Copies = N - 1.
    """
    
    print(" Starting targeted analysis ('components' method)...")

    # --- 1. Load Reference Data (ID -> Sign) ---
    try:
        df_base = pd.read_csv(base_data_path)
        id_to_sign = dict(zip(df_base['ID'], df_base['sign']))
        
        # Denominators: How many times each sign appears total per category
        # Prepare a nested dict: totals[category][sign]
        totals = {}
        
        # For standard categories
        for cat in df_base['category'].unique():
            totals[cat] = df_base[df_base['category'] == cat]['sign'].value_counts().to_dict()
            
        # For 'full' category (entire dataset)
        totals['full'] = df_base['sign'].value_counts().to_dict()
        
    except Exception as e:
        print(f"Error loading base data: {e}")
        return

    # --- 2. Analyze 'components' reports ---
    categories = ['career', 'general', 'love', 'wellness', 'full']
    method = 'components' # Force this method
    
    results_list = []

    for cat in categories:
        filename = f"{cat}_{method}_full_report.csv"
        filepath = os.path.join(reports_folder_path, filename)
        
        if not os.path.exists(filepath):
            print(f"Missing file: {filename}")
            continue
            
        print(f" Analyzing: {filename}")
        
        try:
            df_report = pd.read_csv(filepath)
            
            # Copy counter for this category
            # ex: {'aries': 50, 'taurus': 12...}
            cat_copy_counts = Counter()
            
            # Iterate through clusters
            for _, row in df_report.iterrows():
                if pd.isna(row['IDS']): continue
                
                # Retrieve cluster IDs
                ids_str = str(row['IDS'])
                if not ids_str: continue
                
                cluster_ids = [int(x) for x in ids_str.split(',')]
                
                # Convert IDs -> Signs
                cluster_signs = [id_to_sign.get(i) for i in cluster_ids if i in id_to_sign]
                
                # Count occurrences WITHIN this cluster
                # ex: {'aries': 3, 'leo': 1}
                sign_counts_in_cluster = Counter(cluster_signs)
                
                # Apply rule: Duplicates = n - 1
                for sign, n in sign_counts_in_cluster.items():
                    if n > 1:
                        cat_copy_counts[sign] += (n - 1)
            
            # Calculate percentages for this category
            for sign in df_base['sign'].unique():
                n_total = totals.get(cat, {}).get(sign, 0)
                n_copies = cat_copy_counts.get(sign, 0)
                
                if n_total > 0:
                    rate = (n_copies / n_total) * 100
                else:
                    rate = 0.0
                
                results_list.append({
                    'Category': cat,
                    'Sign': sign,
                    'Total_Horoscopes': n_total,
                    'Detected_Copies': n_copies,
                    'Victimization_Rate': round(rate, 2)
                })
                
        except Exception as e:
            print(f"Error on {filename}: {e}")

    # --- 3. Create Final Table ---
    df_res = pd.DataFrame(results_list)
    
    # Pivot for display (Rows: Signs, Columns: Categories)
    pivot_table = df_res.pivot(index='Sign', columns='Category', values='Victimization_Rate')
    
    # Add an average for sorting
    pivot_table['Average'] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values('Average', ascending=False)
    
    print("\n" + "="*80)
    print("VICTIMIZATION RATE (INTRA-SIGN RECYCLING) - COMPONENTS METHOD")
    print("="*80)
    display(pivot_table.style.background_gradient(cmap='Reds', vmin=0, vmax=100).format("{:.2f}%"))
    
    return df_res

###############################################################################
def plot_small_clusters_distribution(reports_folder_path):
    # Parameters
    categories = ['career', 'general', 'love', 'wellness', 'full']
    methods = ['louvain', 'greedy', 'components'] # Add 'girvan_newman' if necessary
    max_size_x = 10 # Cut off X-axis at 10 as requested

    # Style configuration
    sns.set_theme(style="whitegrid")
    
    # Create one figure per method
    # We could use 3 subplots, but separate figures are more readable here
    
    for method in methods:
        print(f" Generating chart for method: {method.upper()}...")
        
        # 1. Compile data for this method
        data_for_plot = []
        
        for cat in categories:
            filename = f"{cat}_{method}_full_report.csv"
            filepath = os.path.join(reports_folder_path, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, usecols=['Size'])
                    
                    # Keep only small clusters (<= 10)
                    df_small = df[df['Size'] <= max_size_x].copy()
                    
                    if not df_small.empty:
                        df_small['Category'] = cat # For grouping (hue)
                        data_for_plot.append(df_small)
                        
                except Exception as e:
                    pass # Ignore reading errors
        
        if not data_for_plot:
            print(f" No data found for method {method}.")
            continue

        # Merge data
        df_combined = pd.concat(data_for_plot, ignore_index=True)
        
        plt.figure(figsize=(14, 6))
        
        ax = sns.countplot(
            data=df_combined, 
            x='Size', 
            hue='Category', 
            palette='viridis',
            edgecolor='black',
            alpha=0.9
        )
        
        plt.title(f"Small Cluster Size Distribution (1-{max_size_x}) - Method: {method.upper()}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Cluster Size (Number of horoscopes)", fontsize=12)
        plt.ylabel("Number of Clusters (Frequency)", fontsize=12)
        
        plt.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for container in ax.containers:
            ax.bar_label(container, fontsize=8, padding=2)

        plt.tight_layout()
        plt.show()
        
################################################################################
def generate_cohesion_reports(hdf_file_path, threshold=0.5, min_cluster_size=2):
    print(f"Reading file: {hdf_file_path}...")
    try:
        df = pd.read_hdf(hdf_file_path, key='clusters')
    except (FileNotFoundError, KeyError):
        print("Error: File inaccessible or key not found.")
        return

    # Cleaning
    df = df.dropna(subset=['horoscope'])
    counts = df['cluster_label'].value_counts()
    valid_clusters = counts[counts >= min_cluster_size].index
    
    high_cohesion_data = []
    mixed_cohesion_data = []
    
    print(f"Analyzing semantic cohesion on {len(valid_clusters)} clusters (Threshold={threshold})...")
    
    for cluster_id in valid_clusters:
        texts = df[df['cluster_label'] == cluster_id]['horoscope'].tolist()
        
        if len(texts) < 2:
            continue 

        try:
            tfidf = TfidfVectorizer(stop_words='english')
            matrix = tfidf.fit_transform(texts)
            sim_matrix = cosine_similarity(matrix)
            
            upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
            similarities = sim_matrix[upper_triangle_indices]
            
            if len(similarities) == 0:
                continue

            min_sim = np.min(similarities)
            mean_sim = np.mean(similarities)
            
            row_data = {
                'Cluster ID': cluster_id,
                'Size': len(texts),
                'Min Similarity': round(min_sim, 3),
                'Mean Similarity': round(mean_sim, 3),
                'Sample Text': texts[0][:100] + "..."
            }
            
            if min_sim >= threshold:
                high_cohesion_data.append(row_data)
            else:
                mixed_cohesion_data.append(row_data)
                
        except ValueError:
            continue

    cols = ['Cluster ID', 'Size', 'Min Similarity', 'Mean Similarity', 'Sample Text']

    if high_cohesion_data:
        df_high = pd.DataFrame(high_cohesion_data).sort_values('Min Similarity', ascending=False)
    else:
        df_high = pd.DataFrame(columns=cols)

    if mixed_cohesion_data:
        df_mixed = pd.DataFrame(mixed_cohesion_data).sort_values('Min Similarity', ascending=True)
    else:
        df_mixed = pd.DataFrame(columns=cols)
    
    base_name = hdf_file_path.replace('.h5', '')
    
    print("\n" + "="*80)
    print(f"HIGH COHESION CLUSTERS (Min Sim >= {threshold})")
    print(f"   Number of clusters: {len(df_high)}")
    if not df_high.empty:
        display(df_high.head(5))
        df_high.to_csv(f"{base_name}_high_cohesion.csv", index=False)
    else:
        print("   -> No clusters found.")

    print("\n" + "="*80)
    print(f"MIXED COHESION CLUSTERS (Min Sim < {threshold})")
    print(f"   Number of clusters: {len(df_mixed)}")
    if not df_mixed.empty:
        display(df_mixed.head(5))
        df_mixed.to_csv(f"{base_name}_mixed_cohesion.csv", index=False)
    else:
        print("   -> No clusters found.")