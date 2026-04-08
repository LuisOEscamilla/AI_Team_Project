import difflib
from utils import tokenize


class ReviewLogic:
    def __init__(self):
        # Rule-based topic keywords. Keys become predicate names.
        self.rules = {
            'Logistics': ['shipping', 'delivery', 'late', 'arrived', 'delayed'],
            'Quality':   ['material', 'fabric', 'durable', 'broke', 'sturdy'],
            'Fit':       ['size', 'small', 'large', 'tight', 'fit'],
        }

        # Sentiment polarity word lists
        self._positive = ['great', 'love', 'perfect', 'excellent', 'amazing',
                          'beautiful', 'comfortable', 'soft', 'flattering']
        self._negative = ['bad', 'terrible', 'disappointed', 'poor', 'awful',
                          'cheap', 'uncomfortable', 'stiff', 'returned']

        # Negation words that can flip nearby keywords
        self._negators = {'not', 'no', 'never', 'neither', 'nor',
                          "didn't", "doesn't", "isn't", "wasn't",
                          "won't", "wouldn't", "couldn't"}

# Helpers


    def _fuzzy_match(self, word, candidates, cutoff=0.85):
        """
        Return True if `word` is close enough to any candidate keyword.
        Uses difflib SequenceMatcher; no extra dependencies needed.
        """
        matches = difflib.get_close_matches(word, candidates,
                                            n=1, cutoff=cutoff)
        return len(matches) > 0

    def _is_negated(self, tokens, idx, window=2):
        """
        Check whether the token at `idx` is preceded by a negator
        within `window` positions.
        """
        start = max(0, idx - window)
        return any(tokens[j] in self._negators for j in range(start, idx))

# Topic inference


    def infer_topics(self, text):
        """
        Determines which topics (predicates) the review covers.
        Uses fuzzy matching so misspellings still match, and respects negation
        so "no delivery issues" does NOT trigger Logistics.

        Returns a list of topic strings, e.g. ['Fit', 'Quality'].
        Falls back to ['General'] if nothing matches.
        """
        tokens = tokenize(text)
        matched = []
        for topic, keywords in self.rules.items():
            for idx, token in enumerate(tokens):
                if self._fuzzy_match(token, keywords):
                    if not self._is_negated(tokens, idx):
                        matched.append(topic)
                        break          # one match per topic is enough
        return matched if matched else ['General']


# Sentiment – now per-topic


    def _sentiment_score(self, tokens):
        """
        Raw integer score for a token list: +1 per positive word,
        -1 per negative word, with negation inversion.
        """
        score = 0
        for idx, token in enumerate(tokens):
            if self._fuzzy_match(token, self._positive):
                score += -1 if self._is_negated(tokens, idx) else 1
            elif self._fuzzy_match(token, self._negative):
                score += 1 if self._is_negated(tokens, idx) else -1
        return score

    def _label(self, score):
        return 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')

    def sentiment_by_topic(self, text):
        """
        Returns a dict mapping each detected topic to a sentiment label.
        Reviews that mention multiple topics get independent sentiment per topic.

        Example:
            "Love the fabric but the delivery was terrible"
            → {'Quality': 'Positive', 'Logistics': 'Negative'}
        """
        topics = self.infer_topics(text)
        tokens = tokenize(text)

        if topics == ['General']:
            return {'General': self._label(self._sentiment_score(tokens))}

        results = {}
        for topic in topics:
            # Only score the tokens that are near a keyword for this topic
            keywords = self.rules[topic]
            # Find sentence fragments (~5-word windows) around each keyword hit
            topic_tokens = []
            for idx, token in enumerate(tokens):
                if self._fuzzy_match(token, keywords):
                    window_start = max(0, idx - 5)
                    window_end = min(len(tokens), idx + 6)
                    topic_tokens.extend(tokens[window_start:window_end])

            score = self._sentiment_score(topic_tokens) if topic_tokens \
                else self._sentiment_score(tokens)
            results[topic] = self._label(score)

        return results

    def simple_sentiment(self, text):
        """
        Convenience method: single overall sentiment label.
        Kept for backward-compatibility with existing pipeline code.
        """
        tokens = tokenize(text)
        return self._label(self._sentiment_score(tokens))