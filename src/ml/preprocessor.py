"""
Text Preprocessing Pipeline for SupportSense.

Provides both:
1. Basic Preprocessing: Case folding, URL/email/punctuation normalization, contraction expansion, whitespace cleaning.
2. Extended Preprocessing: Basic + POS-aware WordNet lemmatization + intent-preserving stopword filtering.

IMPORTANT DESIGN PRINCIPLE:
Standard stopword lists remove critical negation ('not', 'no', 'never'), interrogative ('why', 'how', 'where'),
and directional/state words ('off', 'up', 'down', 'out', 'again') which directly alter customer intent in banking.
Our stopword filtering selectively preserves all intent-bearing linguistic tokens.
"""

import re
import string
import unicodedata
from typing import List, Optional
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Common English contractions mapping for consistent normalization
CONTRACTIONS_MAP = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "it's": "it is",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "they're": "they are",
    "we're": "we are",
    "there's": "there is",
    "that's": "that is"
}

# Tokens critical to banking customer intent that must NEVER be removed
INTENT_PRESERVED_WORDS = {
    "not", "no", "nor", "never", "none", "cannot", "without", "against",
    "why", "how", "what", "where", "when", "who", "which",
    "off", "out", "up", "down", "in", "on", "into", "over", "under", "again", "back",
    "twice", "still", "yet", "already", "before", "after", "all", "more"
}


def _build_banking_stopwords() -> set:
    """Build a curated stopword set that removes non-informative filler while retaining intent signals."""
    try:
        base_stopwords = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base_stopwords = set(stopwords.words("english"))
    
    # Exclude all intent-bearing tokens from deletion
    curated = base_stopwords - INTENT_PRESERVED_WORDS
    return curated


BANKING_STOPWORDS = _build_banking_stopwords()
LEMMATIZER = WordNetLemmatizer()


def expand_contractions(text: str) -> str:
    """Expands common contractions (e.g., 'can't' -> 'can not')."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS_MAP.keys()) + r')\b', re.IGNORECASE)

    def _replace(match):
        contracted = match.group(0).lower()
        return CONTRACTIONS_MAP.get(contracted, contracted)

    return pattern.sub(_replace, text)


def normalize_unicode(text: str) -> str:
    """Normalizes accented characters and unicode symbols."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")


def clean_text_basic(text: Optional[str]) -> str:
    """
    Basic Text Normalization:
    - Unicode normalization
    - Contraction expansion
    - Lowercasing
    - Stripping URLs, emails, special punctuation
    - Whitespace collapse
    """
    if not text or not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = expand_contractions(text)
    text = text.lower()

    # Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)

    # Replace punctuation (except hyphens in words) with spaces
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Remove isolated hyphens or digits
    text = re.sub(r'\s+-\s+', ' ', text)

    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_extended(text: Optional[str]) -> str:
    """
    Extended Preprocessing:
    - Runs basic normalization
    - Tokenizes text
    - Removes non-intent filler stopwords
    - Applies WordNet Lemmatization on nouns and verbs
    """
    cleaned_basic = clean_text_basic(text)
    if not cleaned_basic:
        return ""

    tokens = cleaned_basic.split()
    processed_tokens: List[str] = []

    for token in tokens:
        # Filter filler words unless preserved
        if token in BANKING_STOPWORDS:
            continue
        
        # Lemmatize verbs then nouns
        lemmatized = LEMMATIZER.lemmatize(token, pos=wordnet.VERB)
        if lemmatized == token:
            lemmatized = LEMMATIZER.lemmatize(token, pos=wordnet.NOUN)
        
        processed_tokens.append(lemmatized)

    result = " ".join(processed_tokens).strip()
    # Fallback to basic clean if all words were filtered out
    return result if result else cleaned_basic


class TextPreprocessor:
    """Scikit-learn compatible preprocessor wrapper."""

    def __init__(self, mode: str = "basic"):
        if mode not in {"basic", "extended"}:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'basic' or 'extended'.")
        self.mode = mode

    def transform_single(self, text: str) -> str:
        if self.mode == "basic":
            return clean_text_basic(text)
        return clean_text_extended(text)

    def transform(self, texts: List[str]) -> List[str]:
        return [self.transform_single(t) for t in texts]

    def __call__(self, text: str) -> str:
        return self.transform_single(text)
