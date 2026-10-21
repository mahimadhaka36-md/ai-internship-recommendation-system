"""
skill_extractor.py
------------------
Explainable, rule-based technical skill extraction module.

Note for college viva:
This module uses a curated dictionary of technical keywords and regex pattern
matching with boundary checks. It is NOT a trained deep learning / NLP model.
It is deterministic, explainable, and allows users to review and edit their skills.
"""

import re
from typing import List, Set

# Comprehensive, curated dictionary of technical skills common in college internships
SKILL_DICTIONARY: List[str] = [
    # Programming Languages
    "python", "java", "c++", "c programming", "c#", "c", "javascript", "typescript",
    "r programming", "r", "php", "ruby", "go", "golang", "swift", "kotlin", "rust", "dart",
    
    # Web & App Development
    "html", "css", "react", "reactjs", "react native", "angular", "vue", "vuejs",
    "node.js", "nodejs", "express", "express.js", "django", "flask", "fastapi",
    "spring", "spring boot", "bootstrap", "tailwind", "jquery", "wordpress",
    "android", "ios", "flutter", "rest api", "restful api", "apis", "api", "graphql",
    
    # Data Science & Machine Learning
    "machine learning", "deep learning", "artificial intelligence", "data science",
    "data analytics", "data analysis", "data visualization", "natural language processing",
    "computer vision", "statistics", "pandas", "numpy", "scikit-learn", "scikit learn",
    "tensorflow", "pytorch", "keras", "opencv", "matplotlib", "seaborn",
    "database testing", "model evaluation", "feature engineering",
    
    # Databases & Big Data
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "firebase",
    "database management systems", "dbms", "data warehousing",
    
    # Business Intelligence & Tools
    "tableau", "power bi", "msexcel", "excel", "google sheets",
    
    # Core CS & Engineering Concepts
    "data structures", "algorithms", "object oriented programming", "oop",
    "operating systems", "computer networks", "software engineering",
    "system design", "problem solving",
    
    # Cloud, DevOps & Systems
    "aws", "azure", "google cloud", "docker", "kubernetes", "linux", "git",
    "github", "ci/cd", "devops", "cloud computing",
    
    # Testing & Cybersecurity
    "testing automation", "selenium", "software testing", "cybersecurity", "networking",
    
    # Design
    "ui/ux design", "ui/ux", "ui design", "ux design", "figma"
]

# Canonical alias mapping for standardization
SKILL_ALIASES = {
    "reactjs": "react",
    "nodejs": "node.js",
    "expressjs": "express.js",
    "vuejs": "vue",
    "scikit learn": "scikit-learn",
    "msexcel": "excel",
    "ms excel": "excel",
    "powerbi": "power bi",
    "c programming": "c",
    "r programming": "r",
    "ui/ux design": "ui/ux",
    "ui design": "ui/ux",
    "ux design": "ui/ux",
    "restful api": "rest api",
    "apis": "api"
}


def normalize_skill_name(skill: str) -> str:
    """Normalize capitalization, whitespace, and known aliases."""
    clean = skill.strip().lower()
    return SKILL_ALIASES.get(clean, clean)


def get_skill_vocab() -> List[str]:
    """Return unique skills sorted by length descending so multi-word skills match first."""
    unique_skills = set(normalize_skill_name(s) for s in SKILL_DICTIONARY)
    # Sort by length descending to prioritize multiword terms like 'machine learning' before 'learning'
    return sorted(unique_skills, key=len, reverse=True)


def extract_skills(text: str) -> List[str]:
    """
    Extract recognized technical skills from raw resume text using word-boundary regexes.
    
    Parameters:
        text (str): Extracted resume plain text.
        
    Returns:
        List[str]: List of unique recognized skills found in the text, sorted alphabetically.
    """
    if not text or not isinstance(text, str):
        return []

    # Normalize whitespace and convert to lowercase for uniform scanning
    cleaned_text = re.sub(r"\s+", " ", text).lower()
    vocab = get_skill_vocab()
    detected: Set[str] = set()

    for skill in vocab:
        # For single-character skills like 'c' or 'r', require strict punctuation/space boundary
        if len(skill) <= 2:
            pattern = rf"(?<![a-z0-9+#.]){re.escape(skill)}(?![a-z0-9+#.])"
        else:
            # Word boundary regex for multi-character terms
            pattern = rf"\b{re.escape(skill)}\b"

        if re.search(pattern, cleaned_text):
            detected.add(skill)

    # Return sorted alphabetical list of recognized skills
    return sorted(list(detected))
