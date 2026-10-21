# KNN Recommendation System on RetailRocket Data Using Graph Data Science

Graph Data Science Midterm Project (UTS) - Universitas Negeri Surabaya

This project applies Graph Data Science to the RetailRocket e-commerce event data. Visitor-product interactions are modeled as a graph in Neo4j, represented as a Vector Space Model (VSM), and then used to build a user-based recommendation system with K-Nearest Neighbors (KNN) and cosine similarity.

## Project Overview

E-commerce data is naturally relational: visitors view, add to cart, and purchase products. In this project, these relationships are stored in Neo4j as a bipartite graph (Visitor - Item), converted into an interaction matrix, and then analyzed to find similarities between users and between items.

## Objectives

- Prepare the RetailRocket event data into two periods: P1 (training) and P2 (testing).
- Import the P1 data into Neo4j using the Python driver.
- Build the User VSM and Item VSM, and compare the User-User and Item-Item projections.
- Compute cosine similarity between items and store it in the graph as `SIMILAR_TO` relationships.
- Generate 3 advertised items for a randomly selected visitor using user-based KNN (k = 5).
- Evaluate the recommendations with Hit@5 and NDCG@5 using the P2 data, for the selected user and as an average over 300 users.

## Dataset

Dataset: RetailRocket E-commerce Dataset (Kaggle). Only `events.csv` is used (2,756,101 rows, 5 columns: `timestamp`, `visitorid`, `event`, `itemid`, `transactionid`), recording interactions from 3 May to 18 September 2015.

| File | Description |
|------|-------------|
| `data/events.csv` | Full event data |
| `data/events_p1.csv` | Events for period P1 (training) |
| `data/events_p2.csv` | Events for period P2 (testing) |

| Period | Range (UTC) | Rows | Users | Purpose |
|--------|-------------|------|-------|---------|
| P1 | 20 June 2015 00:00 - 27 June 2015 (end boundary 28 June 00:00, exclusive) | 155,694 | 83,096 | Building the Neo4j graph and finding KNN neighbors |
| P2 | 28 June 2015 00:00 - 5 July 2015 (end boundary 6 July 00:00, exclusive) | 156,428 | 84,999 | Ground truth for evaluation |

Both periods last 8 days and are consecutive without overlap, so there is no data leakage. Only 3,780 users appear in both periods.

## Methods

| Component | Setting |
|-----------|---------|
| Data representation | Bipartite Visitor - Item graph in Neo4j, then a sparse matrix (`csr_matrix`) |
| Interaction weights | `VIEWED` = 1, `ADDED_TO_CART` = 3, `BOUGHT` = 5 (summed per user-item pair) |
| Recommendation algorithm | User-based KNN (collaborative filtering) |
| Similarity measure | Cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) on the User VSM |
| Number of neighbors | k = 5 (only neighbors with similarity > 0) |
| Recommendation score | Sum over the neighbors of (neighbor similarity x neighbor interaction weight on the item); items the target already touched in P1 are excluded; ties are broken by the smaller item ID |
| Recommendation output | Top-5 ranked list; the top 3 items are the advertised items |
| Evaluation metrics | Hit@5 and NDCG@5 with binary relevance |

**Random user selection (reproducible).** The users that appear in both P1 and P2 are sorted by `visitorId`, and one user is drawn with `np.random.default_rng(42)`. Only users who meet two criteria are eligible: (1) KNN produces 5 recommended items, and (2) the user has at least one item in P2 that was not touched in P1. These criteria are needed because 81.5% of users touch only one item in P1, so many users receive no recommendation at all.

**Ground truth.** The items the user accessed in P2 that were not touched in P1. Items from P1 are never recommended, so counting them as targets would unfairly lower NDCG.

## Technology Stack

Python 3.12, Pandas, NumPy, SciPy, Scikit-learn, NetworkX, Matplotlib, tqdm, Neo4j (Python driver 6.4.1), Cypher, Jupyter Notebook.

## Project Structure

```
UTS-SAINS-DATA-Elvira-Tiara-Suci/
├── UTS_GSD.ipynb
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    ├── events.csv
    ├── events_p1.csv
    └── events_p2.csv
```

## Analytical Workflow

**1. Preliminary tasks.** Set up the virtual environment, learn the Neo4j Python driver (`execute_query`, `MERGE`, `MATCH`), and understand the features of the RetailRocket dataset.

**2. Period definition and data filtering (2.1.1 - 2.1.2).** Filter `events.csv` by UNIX timestamp (milliseconds) into `events_p1.csv` and `events_p2.csv`.

**3. Importing P1 into Neo4j.** The data is imported in batches (5,000 rows) with unique constraints on `Visitor.visitorId` and `Item.itemId`. Relationships are created according to the event type: `VIEWED`, `ADDED_TO_CART`, `BOUGHT`.

**4. Subgraph composition (2.1.3).** Check the number of nodes and relationships using Cypher queries.

**5. User VSM and Item VSM (2.1.4 - 2.1.5).** Build the user x item matrix (and its transpose), analyze sparsity and activity distribution, and visualize samples with NetworkX and subgraphs in Neo4j.

**6. Projection comparison (2.1.6).** Compare the User-User and Item-Item projections in terms of size, density, degree, isolated nodes, and connected components.

**7. Item similarity (2.1.7 - 2.1.8).** Compute the 5 most similar items to reference item 357779 using cosine similarity, store them as `SIMILAR_TO` relationships in Neo4j, and visualize the graph before and after.

**8. Analytics (2.2.1).** Select one eligible target visitor at random (seed 42), find its 5 nearest neighbors with user-based KNN, and generate a top-5 recommendation list, of which the top 3 items are advertised.

## Results

**P1 graph composition**

| Component | Count |
|-----------|-------|
| `Visitor` nodes | 83,096 |
| `Item` nodes | 53,450 |
| `VIEWED` relationships | 150,452 |
| `ADDED_TO_CART` relationships | 3,915 |
| `BOUGHT` relationships | 1,327 |
| Total relationships | 155,694 |

About 96.6% of interactions are `VIEWED`. The total number of relationships equals the number of rows in `events_p1.csv`, so no data was lost during import.

**VSM**

- The user x item matrix has a size of 83,096 x 53,450 with 121,577 filled cells (sparsity 99.9973%).
- 81.5% of users interact with only one item, while 61.38% of items are touched by only one user.
- A dense representation would require about 35.53 GB, while the sparse format needs only about 1.67 MB.

**Projection comparison**

| Aspect | User-User | Item-Item |
|--------|-----------|-----------|
| Number of nodes | 83,096 | 53,450 |
| Connected pairs | 410,094 | 934,700 |
| Density | 0.01188% | 0.06543% |
| Average degree | 9.87 | 34.97 |
| Nodes without neighbors | 26.41% | 49.26% |

The Item-Item projection was chosen for the similarity computation because it is more compact in terms of nodes and gives a stronger similarity signal per node, while User-User is used for the KNN recommendation.

**Item similarity (reference item 357779)**

| Rank | Item ID | Cosine similarity |
|------|---------|-------------------|
| 1 | 66597 | 0.777482 |
| 2 | 242493 | 0.777482 |
| 3 | 403343 | 0.777482 |
| 4 | 257410 | 0.774587 |
| 5 | 304558 | 0.625624 |

**KNN recommendation (k = 5)**

Target: visitor ID 140772. The top-5 list, where the first three items are the advertised items:

| Rank | Item ID | Aggregate score | Advertised |
|------|---------|-----------------|------------|
| 1 | 356315 | 2.383396 | Yes |
| 2 | 453816 | 2.005431 | Yes |
| 3 | 101718 | 1.627467 | Yes |
| 4 | 175668 | 1.627467 | No |
| 5 | 243170 | 1.627467 | No |

Ranks 3, 4, and 5 have the same score, so their order is decided by the tie-breaking rule (smaller item ID first), not by a difference in quality.

**Evaluation**

For the selected user, the ground truth in P2 is a single item (387377), which is not among the five recommended items:

| Metric | Value |
|--------|-------|
| Hit@5 | 0 |
| DCG@5 | 0.0000 |
| IDCG@5 | 1.0000 |
| NDCG@5 | 0.0000 |

Because a single user can only give 0 or 1 for Hit@5, the same metrics were averaged over 300 eligible users (random order, same seed):

| Metric | Mean over 300 users |
|--------|---------------------|
| Hit@5 | 0.0600 |
| NDCG@5 | 0.0301 |

About 6% of the evaluated users (18 of 300) received at least one correct item in the top 5, so the zero for the selected user is the most common outcome rather than an exceptional case. Low values are expected with this dataset: user histories are very short (1.46 items per user on average), the catalog is large (53,450 items), and the ground truth is dominated by `VIEWED` events, while purchases are rare (1,327 of 155,694 relationships in P1).

**Limitations.** The averages only cover users who meet the eligibility criteria, so they are likely more optimistic than the average over all users. The evaluation is not compared against a simple baseline such as recommending the most popular items, so it cannot yet show how much better KNN is than a basic method.

## How to Run

1. Clone the repository:
```
   git clone https://github.com/happ1el/UTS-SAINS-DATA-Elvira-Tiara-Suci.git
   cd UTS-SAINS-DATA-Elvira-Tiara-Suci
```
2. Create and activate a virtual environment:
```
   python -m venv myenv
   myenv\Scripts\activate
```
3. Install the dependencies:
```
   pip install -r requirements.txt
```
4. Start Neo4j, then enter the password when prompted by the notebook (`getpass`). Do not store the password in the repository.
5. Open `UTS_GSD.ipynb` and run all cells in order.

## Author

**Elvira Tiara Suci Tambunan** (24031554213)
Data Science, Universitas Negeri Surabaya

GitHub: [@happ1el](https://github.com/happ1el)