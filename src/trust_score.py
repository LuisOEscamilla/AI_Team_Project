from utils import tokenize, type_token_ratio


class TrustAnalyzer:
    """
    Computes a suspicion score in [0.0, 1.0] with transparent reasons.
    Higher score = more suspicious (less trustworthy).

    Design principle: every heuristic that flags a review must come with
    a human-readable reason so affected users can understand the decision
    (ethics requirement from Presentation I feedback).
    """

    # Weights for each heuristic signal (must sum to 1.0)
    _WEIGHTS = {
        'high_caps':         0.15,   # replaced raw >30% cap check
        'low_info_density':  0.25,
        'low_vocab_diversity':0.20,
        'repetition':        0.20,
        'exclamation_spam':  0.10,
        'single_sentence':   0.10,
    }

    def calculate_suspicion(self, review_text):
        """
        Returns (suspicion_score: float, reasons: list[str]).

        suspicion_score is the weighted sum of triggered heuristics,
        capped at 1.0.
        """
        reasons = []
        raw_scores = {}

        text = str(review_text)
        tokens = tokenize(text)
        word_count = len(tokens)

        # --- Heuristic 1: Extreme capitalisation ---
        # Old threshold (>30%) was too aggressive — a single CAPS word in a
        # short review would trigger it
        # New threshold: >50% AND the review is at least 10 chars long.
        if len(text) >= 10:
            cap_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if cap_ratio > 0.50:
                reasons.append("Very high capitalisation (possible spam or shouting)")
                raw_scores['high_caps'] = 1.0
        raw_scores.setdefault('high_caps', 0.0)

        # --- Heuristic 2: Low information density (very short review) ---
        if word_count < 5:
            reasons.append("Low information density (fewer than 5 words)")
            raw_scores['low_info_density'] = 1.0
        elif word_count < 10:
            raw_scores['low_info_density'] = 0.5
        else:
            raw_scores['low_info_density'] = 0.0

        # --- Heuristic 3: Low vocabulary diversity ---
        # A legitimate review uses varied language; bots/fakes repeat phrases.
        ttr = type_token_ratio(text)
        if word_count >= 10 and ttr < 0.40:
            reasons.append("Low vocabulary diversity (type-token ratio {:.2f})".format(ttr))
            raw_scores['low_vocab_diversity'] = 1.0
        elif word_count >= 10 and ttr < 0.55:
            raw_scores['low_vocab_diversity'] = 0.5
        else:
            raw_scores['low_vocab_diversity'] = 0.0

        # --- Heuristic 4: Word repetition ---
        # If any single word appears more than 30% of the total, flag it.
        if tokens:
            from collections import Counter
            freq = Counter(tokens)
            most_common_ratio = freq.most_common(1)[0][1] / word_count
            if most_common_ratio > 0.30:
                top_word = freq.most_common(1)[0][0]
                reasons.append(
                    "Repetitive content (word '{}' = {:.0%} of review)".format(
                        top_word, most_common_ratio))
                raw_scores['repetition'] = 1.0
            else:
                raw_scores['repetition'] = 0.0
        else:
            raw_scores['repetition'] = 0.0

        # --- Heuristic 5: Exclamation mark spam ---
        excl_count = text.count('!')
        if excl_count >= 3:
            reasons.append("Excessive exclamation marks ({})".format(excl_count))
            raw_scores['exclamation_spam'] = min(1.0, excl_count / 5)
        else:
            raw_scores['exclamation_spam'] = 0.0

        # --- Heuristic 6: Single-sentence review (lacks depth) ---
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        if sentence_count == 1 and word_count >= 5:
            # Only penalise if it's not just inherently short (caught above)
            reasons.append("Single-sentence review (limited context)")
            raw_scores['single_sentence'] = 0.5
        else:
            raw_scores['single_sentence'] = 0.0

        # Weighted sum
        suspicion_score = sum(
            raw_scores[k] * self._WEIGHTS[k]
            for k in self._WEIGHTS
        )

        return round(min(suspicion_score, 1.0), 4), reasons

    def trust_score(self, review_text):
        """Convenience inverse: 1.0 = fully trusted, 0.0 = fully suspicious."""
        suspicion, reasons = self.calculate_suspicion(review_text)
        return round(1.0 - suspicion, 4), reasons