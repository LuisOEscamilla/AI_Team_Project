class TrustAnalyzer:
    def calculate_suspicion(self, review_text):
        """
        Flags potential issues but provides gives a reason to maintain transparency for affected users.
        """
        reasons = []
        # Example Heuristic:Excessive use of caps might be promotional or spammy
        if sum(1 for c in review_text if c.isupper()) / len(review_text) > 0.3:
            reasons.append("High Capitalization")
        
        # Example Heuristic: extremely short reviews lack depth.
        if len(review_text.split()) < 5:
            reasons.append("Low Information Density")
            
        suspicion_score = len(reasons) * 0.5
        return min(suspicion_score, 1.0), reasons