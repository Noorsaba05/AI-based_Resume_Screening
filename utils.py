import re
import spacy
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model (ensure you ran: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")

def extract_text_from_pdf(file_obj):
    """
    Extracts text from a PDF file object using pdfminer.six.
    """
    try:
        text = extract_text(file_obj)
        return text
    except Exception as e:
        return ""

def clean_text(text):
    """
    Applies regex cleaning and spaCy lemmatization.
    1. Removes URLs, Emails, Phone numbers.
    2. Tokenizes and Lemmatizes using spaCy.
    """
    # 1. Regex Cleaning
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', '', text)  # Remove Emails
    text = re.sub(r'[\+\(]?[1-9][0-9.\-\(\)]{8,}[0-9]', '', text)  # Remove Phones
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    # 2. NLP Processing (Lemmatization)
    doc = nlp(text.lower())
    
    # Keep only alpha tokens and remove stop words
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    return " ".join(tokens)

def extract_skills(text):
    """
    A simple rule-based extractor for common tech skills to display in the UI.
    """
    # Common tech keywords to look for (you can expand this list or use EntityRuler)
    skills_db = [
        "python", "java", "sql", "machine learning", "deep learning", 
        "nlp", "tensorflow", "pytorch", "pandas", "react", "node", 
        "aws", "docker", "kubernetes", "c++", "linux", "git"
    ]
    
    found_skills =[]
    text_lower = text.lower()
    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill.capitalize())
            
    return ", ".join(found_skills)

def calculate_similarity(job_desc, resume_texts):
    """
    Calculates TF-IDF vectors and Cosine Similarity.
    
    Args:
        job_desc (str): The clean job description.
        resume_texts (list): List of clean resume strings.
        
    Returns:
        list: Similarity scores for each resume.
    """
    # Create the corpus: Index 0 is JD, Index 1..N are Resumes
    corpus = [job_desc] + resume_texts
    
    vectorizer = TfidfVectorizer(max_features=2000)
    
    # Fit and Transform
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate Cosine Similarity
    # Compare JD (index 0) with all Resumes (index 1 to end)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    return similarity_scores

