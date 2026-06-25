"""
Project 3: AI Recommendation Logic - Tech Stack Recommender
Batch: 2026 | Powered by DecodeLabs

A complete content-based recommendation system that maps user skills
to relevant job roles using TF-IDF and Cosine Similarity.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: DATA INGESTION - RAW SKILLS DATASET
# ============================================
print("="*70)
print("PROJECT 3: AI RECOMMENDATION LOGIC")
print("Tech Stack Recommender - Content-Based Filtering")
print("Powered by DecodeLabs | Batch: 2026")
print("="*70)

# Create the dataset (since we don't have the actual CSV)
# This simulates raw_skills.csv with job roles and required skills
def create_dataset():
    """Create a dataset of job roles with associated skills"""
    
    data = {
        'job_role': [
            # Software Development Roles
            'Full Stack Developer',
            'Backend Developer',
            'Frontend Developer',
            'DevOps Engineer',
            'Mobile App Developer',
            'Game Developer',
            
            # Data Science & AI Roles
            'Data Scientist',
            'Machine Learning Engineer',
            'Data Analyst',
            'Business Intelligence Analyst',
            'AI Research Scientist',
            
            # Cloud & Infrastructure
            'Cloud Architect',
            'Cloud Engineer',
            'Systems Administrator',
            'Network Engineer',
            'Security Engineer',
            
            # Specialized Roles
            'Blockchain Developer',
            'IoT Engineer',
            'Embedded Systems Engineer',
            'Quantitative Analyst',
            'Database Administrator',
            
            # Management Roles
            'Technical Project Manager',
            'Product Manager',
            'Engineering Manager',
            'CTO',
            
            # Additional Roles
            'RPA Developer',
            'QA Engineer',
            'Release Manager',
            'Scrum Master',
            'Solutions Architect'
        ],
        
        'skills': [
            # Software Development
            'Python JavaScript React Node.js MongoDB Express.js HTML CSS Git Docker',
            'Python Java Spring Boot MySQL PostgreSQL Redis Git Microservices AWS',
            'JavaScript React Angular Vue.js HTML CSS TypeScript Bootstrap Git Webpack',
            'Python Docker Kubernetes Jenkins AWS Ansible Terraform CI/CD Linux Bash Git',
            'Kotlin Java Swift React Native Flutter Android iOS Firebase Git REST API',
            'C++ C# Unity Unreal Engine Python Lua Game Design OpenGL DirectX',
            
            # Data Science
            'Python R SQL Machine Learning Deep Learning Statistics Pandas NumPy Scikit-learn TensorFlow',
            'Python R SQL Machine Learning Deep Learning TensorFlow PyTorch Keras NLP Computer Vision',
            'SQL Python Tableau Power BI Excel Data Visualization Statistics Pandas NumPy',
            'SQL Power BI Tableau Excel Data Warehousing ETL Business Intelligence',
            'Python C++ Java Mathematics Deep Learning NLP Computer Vision Research',
            
            # Cloud & Infrastructure
            'AWS Azure GCP Docker Kubernetes Terraform Python Linux Networking Security',
            'AWS Azure GCP Docker Linux Networking Python Bash Ansible Terraform',
            'Linux Windows Active Directory Bash Python Networking Virtualization Storage',
            'Cisco Juniper TCP/IP Routing Switching Firewalls Linux Python Network Security',
            'Security Python C++ Linux Networking Ethical Hacking Firewalls SIEM Compliance',
            
            # Specialized
            'Solidity Rust Python JavaScript Blockchain Smart Contracts Ethereum Cryptography',
            'C C++ Python Embedded Systems Raspberry Pi Arduino IoT MQTT Wireless Sensors',
            'C C++ Assembly Python Embedded Systems RTOS Microcontrollers Linux',
            'Python R SQL Mathematics Statistics Finance Quantitative Analysis',
            'SQL Oracle PostgreSQL MySQL NoSQL Database Design Backup Recovery Performance',
            
            # Management
            'Project Management Scrum Agile JIRA Leadership Communication Planning',
            'Product Strategy UX Design Agile Scrum Market Research Roadmapping',
            'Leadership Agile DevOps Software Architecture Team Management',
            'Technology Strategy Leadership Innovation Business Development',
            
            # Additional
            'Python RPA Automation Anywhere Blue Prism UiPath SQL',
            'Python Selenium Jenkins Docker Automated Testing Postman REST API',
            'CI/CD Jenkins Docker Kubernetes Git Release Management',
            'Scrum Agile JIRA Confluence Team Facilitation Retrospectives',
            'Architecture Cloud DevOps Enterprise Integration'
        ]
    }
    
    return pd.DataFrame(data)

# Load the dataset
df = create_dataset()
print(f"\n📊 DATASET LOADED")
print("-"*50)
print(f"Total Job Roles: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head(10).to_string(index=False))

# ============================================
# STEP 2: DATA PREPROCESSING - TF-IDF VECTORIZATION
# ============================================
print("\n" + "="*70)
print("⚙️ DATA PREPROCESSING: TF-IDF VECTORIZATION")
print("="*70)

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    max_features=100
)

# Fit and transform the skills data
tfidf_matrix = vectorizer.fit_transform(df['skills'])
feature_names = vectorizer.get_feature_names_out()

print(f"\n✅ TF-IDF Vectorization complete!")
print(f"   Vocabulary Size: {len(feature_names)} unique terms")
print(f"   Matrix Shape: {tfidf_matrix.shape}")
print(f"\n   Top 10 features:")
print(f"   {', '.join(feature_names[:10])}...")

# Display TF-IDF matrix as DataFrame for better visualization
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=df['job_role']
)

print(f"\n📊 TF-IDF Matrix Preview (First 5 roles):")
print(tabulate(
    tfidf_df.iloc[:5, :10],
    headers='keys',
    tablefmt='grid',
    floatfmt='.3f'
))

# ============================================
# STEP 3: FUNCTION TO GET USER INPUT
# ============================================
def get_user_input():
    """Collect user skills and preferences"""
    
    print("\n" + "="*70)
    print("👤 USER INPUT - SKILLS COLLECTION")
    print("="*70)
    print("\n📌 Enter your skills and preferences:")
    print("   (These will be mapped to job role recommendations)")
    
    # Collect at least 3 skills as specified
    skills = []
    print("\n🔧 Enter 3 or more technical skills (press Enter after each):")
    
    for i in range(1, 5):
        skill = input(f"   Skill {i}: ").strip()
        if skill:
            skills.append(skill)
        else:
            break
    
    # Ensure we have at least 3 skills
    while len(skills) < 3:
        skill = input(f"   Enter skill {len(skills)+1} (required): ").strip()
        if skill:
            skills.append(skill)
    
    print(f"\n✅ Skills collected: {', '.join(skills)}")
    return skills

# ============================================
# STEP 4: PROCESS USER INPUT
# ============================================
def process_user_input(skills, vectorizer, tfidf_matrix, df):
    """
    Process user input to generate recommendations
    
    Args:
        skills: List of user skills
        vectorizer: TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix of job roles
        df: Original dataset
    
    Returns:
        recommendations: Sorted list of recommended roles
    """
    
    # Combine skills into a single string (same format as dataset)
    user_skills_string = ' '.join(skills)
    
    # Vectorize user skills using the same vectorizer
    user_vector = vectorizer.transform([user_skills_string])
    
    # Calculate cosine similarity between user and all job roles
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Get top recommendations
    df_with_scores = df.copy()
    df_with_scores['similarity_score'] = similarities
    
    # Sort by similarity score
    recommendations = df_with_scores.sort_values(
        'similarity_score', 
        ascending=False
    )
    
    return recommendations

# ============================================
# STEP 5: DISPLAY RECOMMENDATIONS
# ============================================
def display_recommendations(recommendations, top_n=3):
    """
    Display top N recommendations with details
    
    Args:
        recommendations: DataFrame with similarity scores
        top_n: Number of recommendations to show
    """
    
    print("\n" + "="*70)
    print(f"🎯 TOP {top_n} RECOMMENDED JOB ROLES")
    print("="*70)
    
    # Take top N
    top_roles = recommendations.head(top_n)
    
    # Create display table
    display_data = []
    for idx, row in top_roles.iterrows():
        # Get skills as list
        skills_list = row['skills'].split()[:5]  # Top 5 skills
        skills_str = ', '.join(skills_list) + ('...' if len(row['skills'].split()) > 5 else '')
        
        display_data.append([
            row['job_role'],
            f"{row['similarity_score']*100:.2f}%",
            skills_str,
            row['skills']
        ])
    
    # Print table
    headers = ['Job Role', 'Match %', 'Key Skills', 'All Skills']
    print(tabulate(display_data, headers=headers, tablefmt='grid'))
    
    # Show all recommended roles with scores (for analysis)
    print("\n📊 All Job Roles Sorted by Match:")
    print("-"*50)
    
    all_data = []
    for idx, row in recommendations.iterrows():
        all_data.append([
            row['job_role'],
            f"{row['similarity_score']*100:.2f}%"
        ])
    
    # Show only top 10 to avoid clutter
    print(tabulate(all_data[:10], headers=['Job Role', 'Similarity %'], tablefmt='simple'))
    
    # Return top recommendations for visualization
    return top_roles

# ============================================
# STEP 6: VISUALIZATION
# ============================================
def create_visualizations(recommendations, top_n=5, user_skills=None):
    """
    Create visualizations for the recommendation results
    """
    
    # Take top N for visualization
    top_roles = recommendations.head(top_n)
    
    # Figure 1: Bar chart of top recommendations
    plt.figure(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    bars = plt.barh(
        top_roles['job_role'],
        top_roles['similarity_score'] * 100,
        color=colors[:len(top_roles)]
    )
    
    plt.xlabel('Similarity Score (%)', fontsize=12)
    plt.title('Top Job Role Recommendations\nBased on Your Skills', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, top_roles['similarity_score'] * 100):
        plt.text(score + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}%', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('top_recommendations.png', dpi=100, bbox_inches='tight')
    print("✅ Visualization saved as 'top_recommendations.png'")
    
    # Figure 2: Similarity heatmap
    if top_n > 1:
        # Create similarity matrix between top roles
        top_indices = recommendations.head(top_n).index
        top_tfidf = tfidf_matrix[top_indices]
        similarity_matrix = cosine_similarity(top_tfidf)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=top_roles['job_role'],
            yticklabels=top_roles['job_role']
        )
        plt.title('Similarity Between Recommended Job Roles', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('similarity_heatmap.png', dpi=100, bbox_inches='tight')
        print("✅ Visualization saved as 'similarity_heatmap.png'")
    
    # Figure 3: Full distribution of similarities
    plt.figure(figsize=(10, 5))
    all_scores = recommendations['similarity_score'] * 100
    
    plt.hist(all_scores, bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
    plt.xlabel('Similarity Score (%)', fontsize=12)
    plt.ylabel('Number of Job Roles', fontsize=12)
    plt.title('Distribution of Similarity Scores Across All Job Roles', fontsize=14, fontweight='bold')
    plt.axvline(all_scores.mean(), color='red', linestyle='--', label=f'Mean: {all_scores.mean():.1f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('similarity_distribution.png', dpi=100, bbox_inches='tight')
    print("✅ Visualization saved as 'similarity_distribution.png'")
    
    plt.show()

# ============================================
# STEP 7: COLD START HANDLING
# ============================================
def handle_cold_start(skills, df):
    """
    Handle cold start problem by providing fallback recommendations
    
    Args:
        skills: User skills
        df: Original dataset
    
    Returns:
        recommendations: List of fallback recommendations
    """
    
    print("\n⚠️ COLD START DETECTED")
    print("-"*50)
    print("New user or limited data. Using fallback strategy...")
    
    # Strategy 1: If no skills, recommend trending roles
    if not skills or all(s.strip() == '' for s in skills):
        print("   → No skills provided. Recommending trending roles...")
        trending_roles = [
            'Data Scientist',
            'Machine Learning Engineer',
            'Cloud Architect',
            'Full Stack Developer',
            'DevOps Engineer'
        ]
        return df[df['job_role'].isin(trending_roles)]
    
    # Strategy 2: If minimal skills, use keyword matching
    if len(skills) < 2:
        print("   → Limited skills. Using keyword matching...")
        # Simple keyword matching
        matches = []
        for _, row in df.iterrows():
            if any(skill.lower() in row['skills'].lower() for skill in skills):
                matches.append(row)
        
        if matches:
            return pd.DataFrame(matches)
        else:
            # Fallback to trending
            trending_roles = ['Full Stack Developer', 'Data Analyst', 'Cloud Engineer']
            return df[df['job_role'].isin(trending_roles)]
    
    return None

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Main execution function"""
    
    try:
        # Get user input
        user_skills = get_user_input()
        
        # Process recommendations
        recommendations = process_user_input(
            user_skills, 
            vectorizer, 
            tfidf_matrix, 
            df
        )
        
        # Check if we have valid recommendations
        if recommendations['similarity_score'].max() < 0.01:
            print("\n⚠️ Low similarity scores detected. Using fallback...")
            fallback_recommendations = handle_cold_start(user_skills, df)
            if fallback_recommendations is not None:
                recommendations = process_user_input(
                    ['popular', 'trending'],
                    vectorizer,
                    tfidf_matrix,
                    df
                )
        
        # Display recommendations
        top_roles = display_recommendations(recommendations, top_n=5)
        
        # Create visualizations
        create_visualizations(recommendations, top_n=5, user_skills=user_skills)
        
        # Summary
        print("\n" + "="*70)
        print("📌 PROJECT SUMMARY")
        print("="*70)
        print(f"""
    ✅ Recommendation Engine completed successfully!
    
    📊 Dataset: {len(df)} job roles with skills
    🔧 User Skills: {', '.join(user_skills)}
    🎯 Top Match: {top_roles.iloc[0]['job_role']} ({top_roles.iloc[0]['similarity_score']*100:.2f}%)
    📈 Matching Algorithm: TF-IDF + Cosine Similarity
    
    📁 Generated Files:
       - top_recommendations.png (Bar chart of top matches)
       - similarity_heatmap.png (Similarity between top roles)
       - similarity_distribution.png (Score distribution)
    """)
        
        print("="*70)
        print("Thank you for completing Project 3!")
        print("DecodeLabs | Batch 2026")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your input and try again.")

# ============================================
# INTERACTIVE MODE FUNCTIONS
# ============================================

def interactive_demo():
    """
    Interactive demo mode with predefined examples
    """
    print("\n" + "="*70)
    print("🎮 INTERACTIVE DEMO MODE")
    print("="*70)
    
    # Predefined user profiles
    profiles = {
        '1': {
            'name': 'Data Scientist',
            'skills': ['Python', 'Machine Learning', 'Statistics', 'SQL']
        },
        '2': {
            'name': 'Full Stack Developer',
            'skills': ['JavaScript', 'React', 'Node.js', 'MongoDB']
        },
        '3': {
            'name': 'Cloud Engineer',
            'skills': ['AWS', 'Docker', 'Kubernetes', 'Python']
        },
        '4': {
            'name': 'DevOps Engineer',
            'skills': ['Linux', 'Docker', 'CI/CD', 'AWS']
        },
        '5': {
            'name': 'Custom Profile',
            'skills': []
        }
    }
    
    print("\n📋 Available profiles:")
    print("-"*40)
    for key, profile in profiles.items():
        if key != '5':
            print(f"  {key}. {profile['name']}: {', '.join(profile['skills'])}")
    print(f"  {key}. {profiles['5']['name']}: Enter your own skills")
    
    choice = input("\nSelect a profile (1-5): ").strip()
    
    if choice in profiles:
        if choice == '5':
            skills = []
            print("\nEnter your skills (at least 3):")
            for i in range(3):
                skill = input(f"Skill {i+1}: ").strip()
                if skill:
                    skills.append(skill)
            user_skills = skills
        else:
            user_skills = profiles[choice]['skills']
            print(f"\n✅ Using {profiles[choice]['name']} profile")
            print(f"   Skills: {', '.join(user_skills)}")
    else:
        print("Invalid choice. Using default profile...")
        user_skills = ['Python', 'SQL', 'Statistics']
    
    # Process and show recommendations
    recommendations = process_user_input(
        user_skills,
        vectorizer,
        tfidf_matrix,
        df
    )
    
    display_recommendations(recommendations, top_n=5)
    create_visualizations(recommendations, top_n=5, user_skills=user_skills)
    
    return user_skills, recommendations

# ============================================
# FUNCTION TO EXPAND DATASET (for testing)
# ============================================
def expand_dataset(df):
    """Add more job roles and skills to the dataset"""
    
    additional_roles = [
        {
            'job_role': 'AR/VR Developer',
            'skills': 'Unity C# 3D Modeling Computer Vision ARKit ARCore Augmented Reality Virtual Reality'
        },
        {
            'job_role': 'Robotics Engineer',
            'skills': 'Python C++ ROS Robotics Control Systems Computer Vision AI Machine Learning'
        },
        {
            'job_role': 'Bioinformatics Scientist',
            'skills': 'Python R Bioinformatics Genomics Data A
