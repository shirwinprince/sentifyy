import re
import nltk
from nltk.corpus import stopwords

# --- OPTIMIZED DOWNLOAD ---
# This checks if they exist first, so it doesn't re-download constantly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Cache the set for speed
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans text by lowering, removing non-alphabetic characters, 
    and filtering out stop words.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # 3. Remove stop words
    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)