# Institutional Book Intelligence System

Institutional-grade machine learning system for book market intelligence, genre affinity modeling, buyer preference analysis, and hybrid recommendation using content-based filtering, collaborative filtering, clustering, and structural analytics.

---

## Overview

This project builds a full-scale intelligence pipeline to analyze book market behavior and predict cross-book traction using statistical modeling, machine learning, and recommendation algorithms.

The system identifies structural patterns in buyer preferences, predicts which books will gain traction based on genre affinity, and generates hybrid recommendations using both content similarity and collaborative filtering.

It produces institutional-level intelligence outputs, visual analytics, trained models, and a fully automated intelligence report.

---

## Core Capabilities

- Market Structure Intelligence  
- Genre Affinity Modeling  
- Content-Based Recommendation Engine  
- Collaborative Filtering Recommendation Engine  
- Hybrid Recommendation System  
- Buyer Preference Modeling  
- Book Similarity Analysis  
- Market Clustering and Segmentation  
- Anomaly Detection  
- Feature Importance Modeling  
- Automated Institutional Intelligence Report Generation  

---

## Machine Learning Architecture

### Data Processing

- Data normalization using StandardScaler  
- Feature encoding for genre and author  
- Missing value handling and preprocessing  

### Structural Intelligence Layer

- Principal Component Analysis (PCA) for dimensionality reduction  
- KMeans clustering for market segmentation  
- Isolation Forest for anomaly detection  

### Recommendation Layer

#### Content-Based Filtering

Uses cosine similarity on scaled feature space to recommend similar books.

#### Collaborative Filtering

Uses TruncatedSVD latent factor modeling to simulate user-book interaction and identify preference patterns.

#### Hybrid Recommendation Engine

Combines content similarity and collaborative filtering to produce robust recommendations.

### Intelligence Modeling

- Mutual information feature importance analysis  
- Genre affinity matrix generation  
- Market structure mapping  

---

## Dataset

Dataset used:

bestsellers with categories.csv

Features include:

- Book Name  
- Author  
- User Rating  
- Reviews  
- Price  
- Year  
- Genre  

The system models Reviews as the primary traction indicator.

---

## Project Structure

```
institutional-book-intelligence-system/
│
├── Analyzer.py
├── bestsellers with categories.csv
├── README.md
├── LICENSE
│
├── Output/
│   ├── charts/
│   ├── models/
│   ├── recommendations/
│   └── Institutional_Book_Intelligence_Report.pdf
│
└── .vscode/
```

---

## Outputs Generated

### Charts and Visual Intelligence

- PCA projections  
- Cluster visualization  
- Genre affinity heatmaps  
- Traction spectrum analysis  
- Volatility analysis  
- Feature importance visualization  

### Trained Models

- Scaler model  
- PCA model  
- KMeans clustering model  
- Isolation Forest anomaly model  
- SVD collaborative filtering model  

### Recommendation Outputs

- Hybrid recommendation dataset  
- Genre affinity matrix  

### Automated Intelligence Report

Institutional_Book_Intelligence_Report.pdf

---

## Installation

Clone the repository:

```
git clone https://github.com/sagnik10/institutional-book-intelligence-system.git
```

Navigate to project directory:

```
cd institutional-book-intelligence-system
```

Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn scipy reportlab
```

---

## Usage

Run the intelligence pipeline:

```
python Analyzer.py
```

The system will automatically:

- Process the dataset  
- Train intelligence models  
- Generate recommendations  
- Produce charts and analytics  
- Save trained models  
- Generate institutional intelligence report  

---

## Recommendation System Design

Content-Based Filtering uses feature similarity between books including genre, rating, reviews, and price.

Collaborative Filtering simulates buyer interactions and learns latent preference factors using dimensionality reduction.

Hybrid Recommendation combines both approaches to produce robust, production-grade recommendations.

---

## Intelligence Applications

- Book recommendation systems  
- Market demand prediction  
- Genre traction forecasting  
- Customer preference modeling  
- Digital bookstore intelligence  
- Publishing industry analytics  
- Retail intelligence systems  

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- SciPy  
- Matplotlib  
- Seaborn  
- ReportLab  

### Machine Learning Algorithms

- Principal Component Analysis  
- KMeans Clustering  
- Isolation Forest  
- TruncatedSVD  
- Cosine Similarity  
- Mutual Information Regression  

---

## Performance Characteristics

- Fully automated intelligence pipeline  
- Scalable to large datasets  
- Production-ready architecture  
- Institutional-grade analytics output  

---

## License

MIT License

---

## Author

Sagnik Sen  

GitHub:  
https://github.com/sagnik10  

---

## Repository

https://github.com/sagnik10/institutional-book-intelligence-system
