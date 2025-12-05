import math
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import norm

def clean_text(text):

    text = text.lower()

    punctuation_to_remove = string.punctuation.replace("'", "")
    translator = str.maketrans('', '', punctuation_to_remove)

    text_without_punct = text.translate(translator)
    
    return text_without_punct

def getting_idf(input_files):
    N = len(input_files)
    doc_freq = Counter()
    
    #IDF, we only want the number of docs the word is in
    for doc in input_files:
        unique_words_in_doc = set(doc) 
        doc_freq.update(unique_words_in_doc)

    # Computing IDF for every word
    idf = {}
    for word, freq in doc_freq.items():
        idf[word] = math.log2(N / freq)

    return idf
    

def getting_tf(input_files):
    tf_list = []
    for doc in input_files:
        counts = Counter(doc)
        max_count = max(counts.values())
        tf = {word: count/max_count for word, count in counts.items()}
        tf_list.append(tf)
    return(tf_list)


def getting_tf_idf(input_files):
    tf_list = getting_tf(input_files)
    idf_list = getting_idf(input_files)

    tfidf_list = []
    for tf in tf_list:
        tfidf = {word: tf.get(word, 0)*idf_list[word] for word in idf_list}
        tfidf_list.append(tfidf)
    
    return(tfidf_list)
        
    
def stop_words(input_files, threshold, arg_print=True): 

    idf = getting_idf(input_files)

    # Stop words identification
    sorted_idf = sorted(idf.items(), key=lambda x: x[1])

    idf_low_cut_off = math.log2(1 / threshold)
    idf_high_cut_off = math.log2(len(input_files) / 1) # words with this score are only present in one prediction 

    cropped_frequent = [(word, score) for (word, score) in sorted_idf if score < idf_low_cut_off]

    stop_words_frequent = [word for word, score in cropped_frequent]

    cropped_rare = [(word, score) for (word, score) in sorted_idf if score >= idf_high_cut_off]

    stop_words_rare = [word for word, score in cropped_rare]

    if arg_print == True:
        print(f"Most common words (present in more than {100*threshold} % of the dataset):")
        print(stop_words_frequent)
        print(f"Which is a total of {len(stop_words_frequent)} words.\n")

        print(f"Words only present in one prediction over the whole dataset:")
        print(stop_words_rare)
        print(f"Which is a total of {len(stop_words_rare)} words.")


        print(f"Total number of stop words: {len(stop_words_rare) + len(stop_words_frequent)}")


    return(cropped_frequent, stop_words_frequent, cropped_rare, stop_words_rare)

def final_prediction_cleaner(text_prediction, stop_words_set):
    
    cleaned_text = clean_text(text_prediction)
    words = cleaned_text.split()
    
    filtered_words = [word for word in words if word not in stop_words_set]
    
    return " ".join(filtered_words)

# def cosine_similarity_numpy(vec_a, vec_b):
#     #implement cosine similarity between two same-sized vector stored in an numpy array

#     dot_product = np.dot(vec_a, vec_b)

#     norm_product = norm(vec_a) * norm(vec_b)

#     #to avoid dividing by 0
#     if norm_product == 0:
#         return 0.0
    
#     return dot_product / norm_product
# It is computationally acceptable to use the version with loops so let's stick to it


def cosine_similarity_in_house(vec_a, vec_b):
    #implement cosine similarity between two same-sized vector stored in an numpy array

    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors should be of same dimension.")
    
    dot_product = 0
    for i in range(len(vec_a)):
        dot_product += vec_a[i] * vec_b[i]

    #norm of vector a
    sum_of_squares_a = 0
    for elt in vec_a:
        sum_of_squares_a += elt ** 2
    norm_a = math.sqrt(sum_of_squares_a)

    #norm of vector b
    sum_of_squares_b = 0
    for elt in vec_b:
        sum_of_squares_b += elt ** 2
    norm_b = math.sqrt(sum_of_squares_b)

    norm_product = norm_a * norm_b

    #to avoid dividing by 0
    if norm_product == 0:
        return 0 
    
    return(dot_product/norm_product)


def predict_cat(cat_dict, test_vector, categories): 

    max_similarity = -1.0
    predicted_cat = None

    for cat in cat_dict:
        ref_cat = cat[categories]
        ref_vector = cat['cosine_vect']

        similarity = cosine_similarity_in_house(ref_vector, test_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            predicted_cat = ref_cat

    return predicted_cat, max_similarity