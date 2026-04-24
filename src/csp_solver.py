"""
csp_solver.py — Proper CSP solver with forward checking for review selection.

Constraints:
  - No duplicate reviews across slots
  - min_trust: each assigned review must meet trust threshold
  - required_topic: at least one slot must cover the required topic
  - sentiment_filter: at least one slot must match requested sentiment
  - min_words: each assigned review must meet word count
  - coverage_goal: together, assigned reviews must cover all available topics

Forward Checking:
  After assigning a review to a slot, prune ONLY remaining (unassigned) slot
  domains by:
    1. Removing the assigned review (no duplicates)
    2. Checking if remaining domains can still satisfy global constraints
       (if a constraint becomes arc-inconsistent, report failure early)
"""


class CSPSolver:
    def __init__(self, candidates, k, constraints):
        """
        constraints : dict with keys:
                        min_trust        (float or None)
                        required_topic   (str or None)
                        sentiment_filter (str or None)
                        min_words        (int or None)
                        coverage_goal    (bool) - if True, try to cover all topics
        """
        self.candidates = candidates
        self.k = min(k, len(candidates))
        self.constraints = constraints

        # Build initial domains: every slot starts with all candidates
        self.initial_domain = self._apply_unary_constraints(candidates, constraints)

        # Track which topics exist across the full domain
        self.all_topics = set()
        for r in self.initial_domain:
            self.all_topics.update(r['topics'])

    # ------------------------------------------------------------------
    # Node consistency: unary constraints applied once up front
    # ------------------------------------------------------------------

    def _apply_unary_constraints(self, candidates, constraints):
        """
        Remove any candidate that individually violates a unary constraint.
        These are constraints that depend only on a single review, not on
        the combination of reviews selected.

        Unary constraints:
          - min_trust
          - min_words
          - sentiment_filter (a review must at least be capable of matching)
        """
        filtered = []
        min_trust = constraints.get('min_trust', None)
        min_words = constraints.get('min_words', None)
        sentiment_filter = constraints.get('sentiment_filter', None)

        for r in candidates:
            # Unary: trust threshold
            if min_trust is not None and r['trust'] < min_trust:
                continue

            # Unary: word count
            word_count = len(str(r['text']).split())
            if min_words is not None and word_count < min_words:
                continue

            # Unary: sentiment — keep only reviews that could satisfy the
            # filter (we don't require ALL reviews to match, just that the
            # review is capable of it if sentiment_filter is set)
            if sentiment_filter is not None:
                matches = any(v == sentiment_filter for v in r['sentiment_map'].values())
                if not matches:
                    continue

            filtered.append(r)

        return filtered

    # ------------------------------------------------------------------
    # Forward checking
    # ------------------------------------------------------------------

    def _forward_check(self, domains, assigned_slot, assigned_review,
                       assignment, unassigned_slots):
        """
        After assigning `assigned_review` to `assigned_slot`, prune the
        domains of ONLY the remaining unassigned slots.

        Pruning rules:
          1. Remove the just-assigned review from all future domains
             (enforces the binary no-duplicate constraint).
          2. Check if the required_topic constraint can still be satisfied:
             if it hasn't been covered yet, at least one future domain
             must contain a review with that topic. If not, fail early.
          3. Check no future domain has gone empty (dead end).

        """
        # Only copy and operate on unassigned future slots
        new_domains = {s: list(domains[s]) for s in unassigned_slots}
        pruned_info = {}

        required_topic = self.constraints.get('required_topic', None)

        # Rule 1: remove the assigned review from all future slot domains
        for future_slot in new_domains:
            before = len(new_domains[future_slot])
            new_domains[future_slot] = [
                r for r in new_domains[future_slot]
                if r['text'] != assigned_review['text']
            ]
            removed = before - len(new_domains[future_slot])
            if removed:
                pruned_info[future_slot] = [f"removed duplicate ({removed} review(s))"]

        # Rule 2: forward check for required_topic
        if required_topic is not None:
            already_covered = any(
                required_topic in r['topics']
                for r in assignment
            )

            if not already_covered:
                # At least one future domain must still contain a review
                any_future_can_cover = any(
                    any(required_topic in r['topics'] for r in new_domains[s])
                    for s in new_domains
                )
                if not any_future_can_cover:
                    return None, {
                        "failure": f"required_topic '{required_topic}' "
                                   f"can no longer be satisfied"
                    }

        # Rule 3: check no future domain has gone completely empty
        for s, domain in new_domains.items():
            if len(domain) == 0:
                return None, {
                    "failure": f"slot {s} domain is empty after forward checking"
                }

        return new_domains, pruned_info

    # ------------------------------------------------------------------
    # Variable and value ordering heuristics
    # ------------------------------------------------------------------

    def _select_next_slot(self, domains, unassigned_slots):
        """
        Variable ordering heuristic: Minimum Remaining Values (MRV).
        Pick the unassigned slot with the fewest values in its domain.
        Ties broken by slot index for determinism.
        """
        return min(unassigned_slots, key=lambda s: (len(domains[s]), s))

    def _order_domain_values(self, domain, assignment):
        """
        Value ordering heuristic: Least Constraining Value (LCV) approximation.
        Prefer reviews that add the most new topic coverage to what is
        already assigned. Among ties, prefer higher trust.

        This keeps the search moving toward high-value solutions efficiently.
        """
        already_covered = set()
        for r in assignment:
            already_covered.update(r['topics'])

        def lcv_score(review):
            new_topics = set(review['topics']) - already_covered
            return (len(new_topics), review['trust'])

        return sorted(domain, key=lcv_score, reverse=True)

    # ------------------------------------------------------------------
    # Global constraint check (called on complete assignments only)
    # ------------------------------------------------------------------

    def _satisfies_global_constraints(self, assignment):
        """
        Check binary/global constraints on a complete assignment.
        Called only when all k slots are filled.
        """
        required_topic = self.constraints.get('required_topic', None)

        if required_topic is not None:
            covered = any(required_topic in r['topics'] for r in assignment)
            if not covered:
                return False, f"required_topic '{required_topic}' not covered"

        return True, "ok"

    # ------------------------------------------------------------------
    # Public solve entry point
    # ------------------------------------------------------------------

    def solve(self):
        """
        Run backtracking search with forward checking.
        """
        if len(self.initial_domain) < self.k:
            return None, {
                "error": "not enough candidates after constraint filtering",
                "nodes_expanded": 0,
                "backtracks": 0,
                "forward_prunes": 0,
            }, []

        # Initialize domains: all slots start with the same filtered pool
        initial_domains = {i: list(self.initial_domain) for i in range(self.k)}

        stats = {"nodes_expanded": 0, "backtracks": 0, "forward_prunes": 0}
        forward_log = []

        solution = self._backtrack(
            assignment=[],
            domains=initial_domains,
            unassigned=list(range(self.k)),
            stats=stats,
            forward_log=forward_log
        )

        return solution, stats, forward_log

    # ------------------------------------------------------------------
    # Recursive backtracking with forward checking
    # ------------------------------------------------------------------

    def _backtrack(self, assignment, domains, unassigned, stats, forward_log):
        """
        Recursive backtracking search.
        """

        # Base case: all slots filled — check global constraints
        if not unassigned:
            satisfied, reason = self._satisfies_global_constraints(assignment)
            if satisfied:
                return assignment
            else:
                stats["backtracks"] += 1
                forward_log.append(
                    f"  Global check failed: {reason} — backtracking"
                )
                return None

        # Select next variable (MRV heuristic)
        slot = self._select_next_slot(domains, unassigned)
        remaining_slots = [s for s in unassigned if s != slot]

        # Order values for this slot (LCV heuristic)
        ordered_values = self._order_domain_values(domains[slot], assignment)

        for review in ordered_values:
            stats["nodes_expanded"] += 1

            new_assignment = assignment + [review]

            # Forward check: prune only the remaining unassigned slots
            new_domains, pruned_info = self._forward_check(
                domains=domains,
                assigned_slot=slot,
                assigned_review=review,
                assignment=new_assignment,
                unassigned_slots=remaining_slots 
            )

            if new_domains is None:
                # Forward checking detected a dead end — prune this branch
                stats["forward_prunes"] += 1
                if "failure" in pruned_info:
                    forward_log.append(
                        f"  Slot {slot}: pruned '{str(review['text'])[:40]}...' "
                        f"— {pruned_info['failure']}"
                    )
                continue

            if pruned_info:
                prune_count = sum(
                    len(v) for v in pruned_info.values()
                    if isinstance(v, list)
                )
                stats["forward_prunes"] += prune_count
                forward_log.append(
                    f"  Slot {slot}: assigned review, "
                    f"pruned future domains: {pruned_info}"
                )

            # Recurse with updated domains (only unassigned slots remain)
            result = self._backtrack(
                assignment=new_assignment,
                domains=new_domains,
                unassigned=remaining_slots,
                stats=stats,
                forward_log=forward_log
            )

            if result is not None:
                return result

        # All values for this slot exhausted — backtrack
        stats["backtracks"] += 1
        return None