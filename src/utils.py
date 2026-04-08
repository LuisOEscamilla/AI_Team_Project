import re
import pandas as pd


def load_reviews(filepath, text_col='Review Text', sample_n=None, seed=42):
    """
    Load the CSV file and return a cleaned DataFrame.
    Removes missing review text, removes duplicate reviews,
    and can optionally take a smaller random sample.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    # Remove rows where the review text column is missing
    df = df.dropna(subset=[text_col])

    # Remove duplicate reviews based on the review text column
    df = df.drop_duplicates(subset=[text_col])

    # Reset the row index after cleaning
    df = df.reset_index(drop=True)

    # If a sample size is given, take a random sample of the reviews
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=seed).reset_index(drop=True)

    # Return the cleaned DataFrame
    return df


def clean_text(text):
    """
    Clean and normalize review text.
    Makes text lowercase, removes extra spaces,
    and removes most special characters.
    """
    # Convert to string, lowercase it, and remove outer spaces
    text = str(text).lower().strip()

    # Replace non-alphanumeric characters except apostrophes with spaces
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Return the cleaned text
    return text


def tokenize(text):
    """Return the cleaned text as a list of word tokens."""
    # Clean the text first, then split it into words
    return clean_text(text).split()


def type_token_ratio(text):
    """
    Measure vocabulary diversity.
    Returns unique words divided by total words.
    """
    # Turn the text into tokens
    tokens = tokenize(text)

    # Avoid division by zero for empty text
    if not tokens:
        return 0.0

    # Return unique token count divided by total token count
    return len(set(tokens)) / len(tokens)


def dedup_reviews(reviews):
    """
    Remove duplicate reviews from a list of raw review strings.
    Uses cleaned text for comparison.
    """
    # Keep track of cleaned reviews already seen
    seen = set()

    # Store the final unique raw reviews
    unique = []

    # Check each review one by one
    for r in reviews:
        # Create a cleaned version to use as the duplicate-check key
        key = clean_text(r)

        # If this cleaned review has not been seen, keep it
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Return the list of unique reviews
    return unique