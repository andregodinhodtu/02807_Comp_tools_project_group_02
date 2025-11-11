# 02807_Comp_tools_project_group_02

## Members

- André Godinho s253707, s253707@dtu.dk
- Maxence Marbouty s253730, s253730@dtu.dk
- Lucas Jutard s253050, s253050@dtu.dk
- Mathieu Lafitte s253262, s253262@dtu.dk
- Teresio Meda s253212, s253212@dtu.dk

---

## Date

11/11/2025

---

## Idea

> The goal for now is to make data analysis to build a social network that connects people together on a basis that is not on their similarities, but their common interest.

---

## Workflow - plan :

 - Sampling and cleaning of the dataset (@Ljutard2023)
   - input : the whole dataset
   - output : a few shorter datasets, chosen with interest (no missing values, selection of 'users, ...). Same features as before, with selected observations.
   - You will find subsets of different sizes with the location in the [data](./data/) folder. You can also use [Data_Analysis](/notebooks/Data_Analysis.ipynb) notebook to obtain the same subsets without the location. 
 - MinHashing (@Teeresio) --> do whatever bro I don't understand that
 - Computing of frequent itemsets for tweets (@MaxenceMarbouty)
   - input : a dataset sample
   - output : the same dataset with a new column for frequent items found for each tweet
 - Description of the people using frequent itemsets (@andregodinhodtu)
   - input : dataset with frequent items
   - output : matrix of distances between people (counting of the common frequent items between people)
 - Graph design and clustering of people (@matlafENSTA)
   - input : @andregodinhodtu's matrix
   - output : 
     - graph analyze (nodes = people, weighted links = items in common).
     - clusters (Girvan-Newman, Louvain, Spectral Clustering...)

---

## Further Analyze : 

Once we clustered our people based on frequent items analyze, what can be inferred ?
 - are people more connected to people from their homecountries ? (correlation study)
 - do clusters make sense (is there a cluster of people wanting hugs, ...)

---

## DATASET

DATASET : https://www.kaggle.com/datasets/vivekchary/sentiment-with-16-million-tweets-with-locations

Content :
datetime (str) | user ID (str) | tweet content (str) | location (str)


## Detailed Explanation

### Graph Clustering
First approach:
The graph is first obtained by setting a */threshold*/ on the number of words in common connecting people: if two people tweeted similar */meaningful/* words more than */threshold/* times, they are considered as connected, otherwise no. 
Second approach:
Connect every users that once tweeted the same meaningful word(s). Connections between users have a weight equal to the number of word tweeted by both users: the result is a weighted graph.

NB: */meaningful/* words