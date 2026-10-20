# KNN Recommendation System on RetailRocket Data Using Graph Data Science

Graph Data Science Midterm Project (UTS) - Universitas Negeri Surabaya

This project applies Graph Data Science to the RetailRocket e-commerce event data. Visitor-product interactions are modeled as a graph in Neo4j, represented as a Vector Space Model (VSM), and then used to build a recommendation system based on K-Nearest Neighbors (KNN) with cosine similarity.

## Project Overview

E-commerce data is naturally relational: visitors view, add to cart, and purchase products. In this project, these relationships are stored in Neo4j as a bipartite graph (Visitor - Item), converted into an interaction matrix, and then analyzed to find similarities between users and between items.

## Objectives

- Prepare the RetailRocket event data into two periods: P1 (training) and P2 (testing).
- Import the P1 data into Neo4j using the Python driver.
- Build the User VSM and Item VSM, and compare the User-User and Item-Item projections.
- Compute cosine similarity between items and store it in the graph as `SIMILAR_TO` relationships.
- Generate 3 item recommendations for a target visitor using KNN (k = 5).
- Evaluate the recommendations with Hit@5 and NDCG@5 using the P2 data.

## Dataset

Dataset: RetailRocket E-commerce Dataset (Kaggle). Only `events.csv` is used (2,756,101 rows, 5 columns: `timestamp`, `visitorid`, `event`, `itemid`, `transactionid`), recording interactions from 3 May to 18 September 2015.

| File | Description |
|------|-------------|
| `data/events.csv` | Full event data |
| `data/events_p1.csv` | Events for period P1 (training) |
| `data/events_p2.csv` | Events for period P2 (testing) |

| Period | Range (UTC) | Purpose |
|--------|-------------|---------|
| P1 | 20 June 2015 00:00 - 27 June 2015 (end boundary 28 June 00:00, exclusive) | Building the Neo4j graph and finding KNN neighbors |
| P2 | 28 June 2015 00:00 - 5 July 2015 (end boundary 6 July 00:00, exclusive) | Ground truth for evaluation |

Both periods last 8 days and are consecutive without overlap, so there is no data leakage.

## Methods

| Component | Setting |
|-----------|---------|
| Data representation | Bipartite Visitor - Item graph in Neo4j, then a sparse matrix (`csr_matrix`) |
| Interaction weights | `VIEWED` = 1, `ADDED_TO_CART` = 3, `BOUGHT` = 5 (summed per user-item pair) |
| Recommendation algorithm | User-based KNN (collaborative filtering), `NearestNeighbors` |
| Similarity measure | Cosine (cosine distance in KNN, cosine similarity for item similarity) |
| Number of neighbors | k = 5 |
| Number of recommendations | 3 items |
| User filter | Only users with at least 3 distinct items (nnz >= 3) |
| Recommendation score | Sum of the five neighbors' interaction weights per item; items already accessed by the target are excluded |
| Evaluation metrics | Hit@5 and NDCG@5 |

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

**8. Analytics (2.2).** Filter active users, randomly select one target visitor (seed 42), find the 5 nearest neighbors with KNN, and generate 3 item recommendations.

**9. Evaluation (2.2.2).** Compute Hit@5 and NDCG@5 against the items accessed by the target in P2.

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

**KNN recommendation (k = 5)**

Target: visitor ID 121147 (4 items in P1). The top three recommended items:

| Rank | Item ID | Aggregate score |
|------|---------|-----------------|
| 1 | 48030 | 18.0 |
| 2 | 125660 | 13.0 |
| 3 | 308053 | 12.0 |

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