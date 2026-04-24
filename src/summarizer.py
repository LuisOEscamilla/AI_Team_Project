"""
summarizer.py  —  Review-of-Reviews main pipeline

Pipeline order:
    load_reviews  →  logic_engine  →  trust_score
        →  CSP solver (forward checking)  →  A* search  →  print summary

CSP Role:  Finds a valid assignment of k review slots satisfying all
           user-defined hard constraints, using backtracking + forward checking.

A* Role:   Within the CSP solution space, optimizes the selected subset
           for maximum topic coverage and trust using admissible heuristic.
"""

import heapq
import os
import pandas as pd
import kagglehub
from utils import load_reviews
from logic_engine import ReviewLogic
from trust_score import TrustAnalyzer
from csp_solver import CSPSolver


# -----------------------------------------------------------------------
# Step 1: Annotate each review with topics, sentiment, and trust
# -----------------------------------------------------------------------

def annotate(df, text_col='Review Text'):
    """
    Adds columns to the DataFrame:
      - topics         : list[str]
      - sentiment_map  : dict[str, str]
      - trust          : float in [0, 1]
      - trust_reasons  : list[str]
    """
    logic = ReviewLogic()
    trust_analyzer = TrustAnalyzer()

    topic_list, sentiment_list, trust_list, reason_list = [], [], [], []

    for text in df[text_col]:
        topics = logic.infer_topics(text)
        sentiment_map = logic.sentiment_by_topic(text)
        trust, reasons = trust_analyzer.trust_score(text)

        topic_list.append(topics)
        sentiment_list.append(sentiment_map)
        trust_list.append(trust)
        reason_list.append(reasons)

    df = df.copy()
    df['topics'] = topic_list
    df['sentiment_map'] = sentiment_list
    df['trust'] = trust_list
    df['trust_reasons'] = reason_list
    return df


# -----------------------------------------------------------------------
# Step 2: CSP solver — replaces the old filter-only approach
# -----------------------------------------------------------------------

def apply_csp(candidates, k, constraints):
    """
    Use the CSPSolver (backtracking + forward checking) to find a valid
    assignment of k review slots satisfying all hard constraints.

    Returns
    -------
    solution     : list of review dicts (valid assignment), or None
    stats        : search statistics dict
    forward_log  : list of forward-checking decisions
    csp_report   : human-readable summary string
    """
    solver = CSPSolver(candidates, k, constraints)

    domain_size = len(solver.initial_domain)
    solution, stats, forward_log = solver.solve()

    csp_report = (
        f"CSP Solver (backtracking + forward checking):\n"
        f"  Candidates after unary constraint filtering: {domain_size} / {len(candidates)}\n"
        f"  Slots to fill: {k}\n"
        f"  Nodes expanded: {stats.get('nodes_expanded', 0)}\n"
        f"  Backtracks: {stats.get('backtracks', 0)}\n"
        f"  Forward prunes: {stats.get('forward_prunes', 0)}\n"
        f"  Constraints applied: {constraints}\n"
        f"  Result: {'Valid assignment found' if solution else 'No solution found'}"
    )

    return solution, stats, forward_log, csp_report


# -----------------------------------------------------------------------
# Step 3: A* search — optimizes the CSP solution for coverage + trust
#
# Admissibility argument:
#   g(n) = topic_weight * (covered_topics / total_topics)
#          + trust_weight * avg_trust_so_far
#
#   h(n) = upper bound on remaining gain
#         = trust_weight * (remaining_slots / k)
#           + topic_weight * (uncovered_topics / total_topics)
# -----------------------------------------------------------------------

def _coverage(selected_topics):
    all_topics = set()
    for t in selected_topics:
        all_topics.update(t)
    return len(all_topics)


def _admissible_heuristic(remaining_reviews, remaining_slots, k,
                           uncovered_topics, total_topics,
                           topic_weight, trust_weight):
    """
    Admissible upper bound on remaining gain.

    Assumes:
      - All remaining reviews have trust = 1.0 (best case)
      - All uncovered topics can still be covered
    """
    if remaining_slots == 0:
        return 0.0

    max_trust_gain = trust_weight * (remaining_slots / k)
    max_topic_gain = topic_weight * (len(uncovered_topics) / max(total_topics, 1))

    return max_trust_gain + max_topic_gain


def astar_search(candidate_reviews, k=5, topic_weight=0.7, trust_weight=0.3,
                 beam_width=120, max_expansions=4000):
    """
    Beam-limited A* search over review subsets.

    Objective:
        g(n) = topic_weight * coverage_ratio + trust_weight * avg_trust

    Heuristic h(n) is admissible (see module docstring above).

    Note: beam_width limits memory at the cost of optimality guarantee.
    In practice for k<=10 and typical review pools this finds the optimum.
    """
    n = len(candidate_reviews)
    if n == 0:
        return []

    k = min(k, n)

    logic = ReviewLogic()
    total_topics = max(1, len(logic.rules))

    # Compute all topics present in candidates
    all_present_topics = set()
    for r in candidate_reviews:
        all_present_topics.update(r['topics'])
    total_topics = max(1, len(all_present_topics))

    # heap item: (-f, g, selected_tuple, remaining_tuple)
    initial_remaining = tuple(range(n))
    initial_h = _admissible_heuristic(
        remaining_reviews=candidate_reviews,
        remaining_slots=k,
        k=k,
        uncovered_topics=all_present_topics,
        total_topics=total_topics,
        topic_weight=topic_weight,
        trust_weight=trust_weight
    )
    heap = [(-initial_h, 0.0, tuple(), initial_remaining)]

    best_solution = None
    best_score = -1.0
    visited = {}
    expansions = 0

    while heap and expansions < max_expansions:
        # Beam limit: keep only best frontier states
        if len(heap) > beam_width:
            heap = heapq.nsmallest(beam_width, heap)
            heapq.heapify(heap)

        neg_f, g, selected, remaining = heapq.heappop(heap)
        expansions += 1

        state_key = selected
        if state_key in visited and visited[state_key] >= g:
            continue
        visited[state_key] = g

        if len(selected) == k:
            if g > best_score:
                best_score = g
                best_solution = selected
            continue

        if not remaining:
            continue

        # Compute currently covered topics
        current_topic_set = set()
        for i in selected:
            current_topic_set.update(candidate_reviews[i]['topics'])

        remaining_list = list(remaining)

        # Score and sort next moves
        scored_next = []
        for idx in remaining_list:
            new_topics = set(current_topic_set)
            new_topics.update(candidate_reviews[idx]['topics'])
            added_topics = len(new_topics) - len(current_topic_set)
            local_score = candidate_reviews[idx]['trust'] + 0.2 * added_topics
            scored_next.append((local_score, idx))

        scored_next.sort(reverse=True)

        # Expand top moves only
        for _, idx in scored_next[:10]:
            new_selected = tuple(sorted(selected + (idx,)))
            new_remaining = tuple(r for r in remaining_list if r != idx)

            selected_topics = [candidate_reviews[s]['topics'] for s in new_selected]
            cov = _coverage(selected_topics) / total_topics
            avg_trust = sum(candidate_reviews[s]['trust'] for s in new_selected) / len(new_selected)
            new_g = topic_weight * cov + trust_weight * avg_trust

            uncovered = all_present_topics - set().union(*[candidate_reviews[s]['topics'] for s in new_selected])
            remaining_slots = k - len(new_selected)
            remaining_reviews_data = [candidate_reviews[r] for r in new_remaining]

            h = _admissible_heuristic(
                remaining_reviews=remaining_reviews_data,
                remaining_slots=remaining_slots,
                k=k,
                uncovered_topics=uncovered,
                total_topics=total_topics,
                topic_weight=topic_weight,
                trust_weight=trust_weight
            )

            heapq.heappush(heap, (-(new_g + h), new_g, new_selected, new_remaining))

    if best_solution is None:
        sorted_idx = sorted(range(n), key=lambda i: candidate_reviews[i]['trust'], reverse=True)
        best_solution = tuple(sorted(sorted_idx[:k]))

    return [candidate_reviews[i] for i in best_solution]


# -----------------------------------------------------------------------
# Step 4: Pretty-print the summary
# -----------------------------------------------------------------------

def print_summary(selected_reviews, csp_report, forward_log=None):
    print("\n" + "=" * 60)
    print("REVIEW SUMMARY")
    print("=" * 60)
    print(csp_report)

    if forward_log:
        print("\nForward Checking Log (first 10 entries):")
        for line in forward_log[:10]:
            print(line)

    print(f"\nSelected {len(selected_reviews)} reviews via A* search:\n")

    for i, r in enumerate(selected_reviews, 1):
        text_preview = str(r['text'])[:120].replace('\n', ' ')
        print(f"  [{i}] Topics:    {r['topics']}")
        print(f"      Sentiment: {r['sentiment_map']}")
        print(f"      Trust:     {r['trust']:.2f}")
        if r.get('trust_reasons'):
            print(f"      Flags:     {'; '.join(r['trust_reasons'])}")
        print(f"      Preview:   \"{text_preview}...\"")
        print()

    all_topics = set()
    for r in selected_reviews:
        all_topics.update(r['topics'])
    print(f"  Topic coverage: {sorted(all_topics)}")
    print("=" * 60)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':
    TEXT_COL = 'Review Text'

    print("Downloading / locating dataset with kagglehub...")
    dataset_path = kagglehub.dataset_download("nicapotato/womens-ecommerce-clothing-reviews")
    DATA_PATH = os.path.join(dataset_path, "Womens Clothing E-Commerce Reviews.csv")

    print(f"Dataset folder: {dataset_path}")
    print(f"CSV path: {DATA_PATH}")

    print("Loading reviews...")
    df = load_reviews(DATA_PATH, text_col=TEXT_COL, sample_n=40)

    print("Annotating with topics, sentiment, and trust scores...")
    df = annotate(df, text_col=TEXT_COL)

    constraints = {
        'min_trust': 0.55,
        'required_topic': None,
        'sentiment_filter': None,
        'min_words': 8,
        'coverage_goal': True,
    }

    k = 5

    print("Running CSP solver (backtracking + forward checking)...")
    candidates = [
        {
            'text': row[TEXT_COL],
            'topics': row['topics'],
            'sentiment_map': row['sentiment_map'],
            'trust': row['trust'],
            'trust_reasons': row['trust_reasons'],
        }
        for _, row in df.iterrows()
    ]

    solution, stats, forward_log, csp_report = apply_csp(candidates, k, constraints)

    if solution is None:
        print("CSP found no valid solution. Relaxing constraints and falling back to A*.")
        solution = candidates

    print("Running A* search for optimal review subset...")
    selected = astar_search(
        solution,
        k=k,
        topic_weight=0.7,
        trust_weight=0.3,
        beam_width=120,
        max_expansions=4000
    )

    print_summary(selected, csp_report, forward_log)