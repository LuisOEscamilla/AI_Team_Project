# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("nicapotato/womens-ecommerce-clothing-reviews")

# print("Path to dataset files:", path)

import pandas as pd

class ReviewLogic:
    def __init__(self):
        # Rule-based logic for topic categorization
        self.rules = {
            'Logistics': ['shipping', 'delivery', 'late', 'arrived', 'delayed'],
            'Quality': ['material', 'fabric', 'durable', 'broke', 'sturdy'],
            'Fit': ['size', 'small', 'large', 'tight', 'fit']
        }

    def infer_topics(self, text):
        """Determines topic relevance using rule-based logic."""
        text = str(text).lower()
        topics = [topic for topic, keywords in self.rules.items() 
                  if any(word in text for word in keywords)]
        return topics if topics else ['General']

    def simple_sentiment(self, text):
        """Basic sentiment logic: searching for polarity keywords."""
        positive = ['great', 'love', 'perfect', 'excellent']
        negative = ['bad', 'terrible', 'disappointed', 'poor']
        text = str(text).lower()
        
        score = sum(1 for w in positive if w in text) - sum(1 for w in negative if w in text)
        return 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')