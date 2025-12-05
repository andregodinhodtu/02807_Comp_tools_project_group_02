# 02807_Comp_tools_project_group_02

## Members

- André Godinho s253707, s253707@dtu.dk
- Maxence Marbouty s253730, s253730@dtu.dk
- Lucas Jutard s253050, s253050@dtu.dk
- Mathieu Lafitte s253262, s253262@dtu.dk
- Teresio Meda s253212, s253212@dtu.dk

---

## Idea / Abstarct

> The goal of the project is to analyze the semantic structure and uniqueness of daily horoscopes, determining whether they contain truly distinctive content or rely on repetitive and generic language. By applying computational text analysis methods such as TF-IDF, MinHashing, and Frequent Itemset Mining alongside graph-based clustering techniques like the Louvain and Greedy algorithms, the study measures textual redundancy and explores whether each zodiac sign’s vocabulary reflects meaningful differences or follows random patterns. Ultimately, the project aims to challenge the specificity, authenticity, and scientific validity of astrological forecasting through a rigorous quantitative analysis of language use.

---

## Workflow - plan :

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

## How to create new env 
```
python -m venv venv
```
(windows)
```
venv\Scripts\activate
```
(Mac/Linux)
```
source venv/bin/activate
```
Then, you need to install requirements:
```
pip install -r requirements.txt
```
Well done !
