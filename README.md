# 💎 Sentify Luxe | Enterprise Sentiment Intelligence

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?logo=chainlink)
![License](https://img.shields.io/badge/license-MIT-green)
![Groq](https://img.shields.io/badge/Powered%20By-Groq-orange)

**Sentify Luxe** is a high-performance analytics dashboard...

💎 Sentify Luxe | Enterprise Sentiment Intelligence
Sentify Luxe is a high-performance analytics dashboard that transforms raw customer feedback into strategic business assets. By merging traditional Machine Learning with Large Language Models (LLMs), it provides a dual-layer approach to understanding the "why" behind customer sentiment.

🎯 Target Audience
Product Managers: Identify feature requests and friction points.

CX Teams: Detect churn risk and high-value promoters instantly.

Marketing Strategists: Extract testimonials from high-confidence positive reviews.

Data Analysts: Automate the path from raw CSV to executive-ready reporting.

✨ Key Features
1. Dual-Engine Intelligence
ML Classifier: High-speed sentiment tagging (Positive/Neutral/Negative) using Scikit-Learn (TF-IDF + Logistic Regression).

AI Analyst: An integrated RAG-capable chatbot powered by Groq (Llama 3.3/3.1) for deep-dive data interrogation.

2. Executive Visualizations
Sunburst Composition: View the relationship between sentiment and review depth (Detailed vs. Short).

Brand Health Gauge: Real-time KPI tracking of overall brand positivity.

Semantic Pattern Analysis: Dynamic word clouds that filter by sentiment segment.

3. Automated Insight Mining
Feature Request Detection: Automatically flags "Wishlist" items and "Potential Leads" for product development.

Scientific Audit: Heatmaps and histograms analyzing model confidence and data distribution.

4. Enterprise-Ready UI
Luxe Theme Engine: Smooth toggle between "Light" and "Dark" modes with custom-designed CSS metric cards.

Data Portability: Export fully processed reports with confidence scores as CSV for external use.

🛠️ Tech Stack
Frontend: Streamlit

NLP/AI: LangChain, Groq API, [HuggingFace Embeddings]

Machine Learning: Scikit-Learn, Joblib

Vector DB: ChromaDB

Visualization: Plotly, WordCloud, Matplotlib

🚀 Installation & Setup

1. Clone the Repository

Bash

git clone https://github.com/shirwinprince/sentifyy.git

cd sentifyy

2. Set Up Environment Variables
   
Create a .env file in the root directory and add your Groq API Key:

Code snippet

GROQ_API_KEY=your_lp_api_key_here

3. Install Dependencies
   
Bash

pip install -r requirements.txt


4. Project Structure
Ensure your directory looks like this:

## 📁 Project Structure

Ensure your directory looks like this:

```
sentifyy/
├── .env                     # Secret API keys (GROQ_API_KEY, etc.)
├── .gitignore               # Files to ignore (venv, .env, __pycache__)
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
├── demo.py                  # Main Streamlit application code
│
├── model/                   # Folder for ML assets
│   ├── sentiment_model.pkl  # Trained Logistic Regression model
│   └── tfidf.pkl            # TF-IDF Vectorizer
│
├── utils/                   # Helper scripts
│   ├── __init__.py          # Makes 'utils' a Python package
│   └── preprocess.py        # clean_text() function logic
│
├── vectorstore/             # Persistent ChromaDB storage
│   └── (Auto-generated files after running the RAG logic)
│
├── assets/                  # UI elements and images
│   ├── l.png                # Logo used in sidebar
│   ├── sa.png               # Empty state illustration
│   └── banner.png           # README banner
│
└── data/                    # Optional sample data
    └── sample_reviews.csv
```

    
5. Run the Application
   
Bash

streamlit run demo.py

📊 Data Requirements
The application expects a .csv file with a column named "Comment".

Input: Raw customer reviews.

Output: Sentiment labels, Confidence scores, and Review depth analysis



<img width="1892" height="871" alt="Screenshot 2026-02-01 195133" src="https://github.com/user-attachments/assets/0f7c755d-ee80-46bd-b7a1-d31943de663e" />

<img width="1910" height="848" alt="Screenshot 2026-02-01 195207" src="https://github.com/user-attachments/assets/42124485-ffc2-4bf2-98aa-616a50c88a73" />

<img width="1919" height="857" alt="Screenshot 2026-02-01 195244" src="https://github.com/user-attachments/assets/88e84600-5b27-4c6f-9a33-0200629f892a" />

<img width="1910" height="870" alt="Screenshot 2026-02-01 195258" src="https://github.com/user-attachments/assets/f6aaa054-4f16-408f-8cd6-e7f74071e014" />

<img width="1908" height="886" alt="Screenshot 2026-02-01 195315" src="https://github.com/user-attachments/assets/3bafa6a8-5ac7-43bd-be28-afaa57669e36" />
