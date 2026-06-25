# Project 3: AI Recommendation Logic

Batch: 2026 | Powered by DecodeLabs

## 📋 Overview

A content-based recommendation system that maps user skills to relevant job roles using TF-IDF vectorization and Cosine Similarity. This project demonstrates the complete pipeline from user input to intelligent recommendations.

## 🎯 Project Goals

- Build a content-based filtering recommendation engine
- Implement TF-IDF for feature extraction
- Use Cosine Similarity for matching
- Handle the Cold Start problem
- Provide visual feedback and analysis


## 📊 Dataset

**Job Roles Dataset**
- Total Roles: 30
- Features: Job Title + Required Skills
- Skills per role: 5-10 technical skills

### Example Roles:
| Job Role | Required Skills |
|----------|-----------------|
| Data Scientist | Python, R, SQL, Machine Learning, Statistics |
| Full Stack Developer | JavaScript, React, Node.js, MongoDB, HTML |
| Cloud Architect | AWS, Azure, GCP, Docker, Kubernetes |
| DevOps Engineer | Linux, Docker, Kubernetes, Jenkins, AWS |

## 🚀 How to Run

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd project3-recommendation

# Install dependencies
pip install -r requirements.txt
