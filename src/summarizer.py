"""
summarizer.py  —  Review-of-Reviews main pipeline

Pipeline order:
    load_reviews  →  logic_engine  →  trust_score
        →  CSP filter  →  A* search  →  print summary
"""

import heapq
import os
import pandas as pd
import kagglehub
from utils import load_reviews
from logic_engine import ReviewLogic
from trust_score import TrustAnalyzer


# Step 1: Annotate each review with topics, sentiment, and trust

def annotate(df, text_col='Review Text'):
    """
    Adds three columns to the DataFrame:
      - topics         : list[str] – detected topic predicates
      - sentiment_map  : dict[str, str] – per-topic sentiment
      - trust          : float – trust score in [0, 1] (higher = more trusted)
      - trust_reasons  : list[str] – human-readable flags
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


# Step 2: CSP filter — user-defined hard constraints


def apply_constraints(df, constraints, text_col='Review Text'):
    """
    Filter the annotated DataFrame to only rows that satisfy ALL constraints.
    """
    original_len = len(df)
    mask = pd.Series([True] * len(df), index=df.index)

    min_trust = constraints.get('min_trust', None)
    if min_trust is not None:
        mask &= df['trust'] >= min_trust

    required_topic = constraints.get('required_topic', None)
    if required_topic:
        mask &= df['topics'].apply(lambda t: required_topic in t)

    sentiment_filter = constraints.get('sentiment_filter', None)
    if sentiment_filter:
        def matches_sentiment(smap):
            return any(v == sentiment_filter for v in smap.values())
        mask &= df['sentiment_map'].apply(matches_sentiment)

    min_words = constraints.get('min_words', None)
    if min_words is not None:
        mask &= df[text_col].apply(lambda t: len(str(t).split()) >= min_words)

    filtered = df[mask].reset_index(drop=True)
    report = (
        f"CSP: {original_len} reviews → {len(filtered)} passed "
        f"({original_len - len(filtered)} filtered out). "
        f"Constraints applied: {constraints}"
    )
    return filtered, report


# Step 3: Faster A* search for optimal review subset


def _coverage(selected_topics):
    """Distinct topic count across a set of reviews."""
    all_topics = set()
    for t in selected_topics:
        all_topics.update(t)
    return len(all_topics)


def _heuristic(remaining_reviews):
    """
    Heuristic: average trust of remaining reviews.
    """
    if not remaining_reviews:
        return 0.0
    return sum(r['trust'] for r in remaining_reviews) / len(remaining_reviews)


def _candidate_priority(review):
    """
    Simple score for pruning before search.
    Higher trust and more topics = more useful.
    """
    return review['trust'] + 0.15 * len(review['topics'])


def astar_search(candidate_reviews, k=5, topic_weight=0.7, trust_weight=0.3,
                 beam_width=120, max_candidates=25, max_expansions=4000):
    """
    Faster beam-limited A* search.

    Key speed fixes:
      - prune candidate pool first
      - keep only top `beam_width` frontier states
      - stop after `max_expansions`
    """
    n = len(candidate_reviews)
    if n == 0:
        return []

    # ---------- prune candidates first ----------
    candidate_reviews = sorted(
        candidate_reviews,
        key=_candidate_priority,
        reverse=True
    )[:max_candidates]

    n = len(candidate_reviews)
    k = min(k, n)

    topic_rule_count = max(1, len(ReviewLogic().rules))

    # heap item: (-f, g, selected_tuple, remaining_tuple)
    initial_remaining = tuple(range(n))
    init_h = _heuristic(candidate_reviews) * trust_weight
    heap = [(-init_h, 0.0, tuple(), initial_remaining)]

    best_solution = None
    best_score = -1.0
    visited = {}
    expansions = 0

    while heap and expansions < max_expansions:
        # beam limit: keep only best frontier states
        if len(heap) > beam_width:
            heap = heapq.nsmallest(beam_width, heap)
            heapq.heapify(heap)

        neg_f, g, selected, remaining = heapq.heappop(heap)
        f = -neg_f
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

        current_topics = [candidate_reviews[i]['topics'] for i in selected]
        current_topic_set = set()
        for topics in current_topics:
            current_topic_set.update(topics)

        remaining_list = list(remaining)

        # expand better next moves first
        scored_next = []
        for idx in remaining_list:
            new_topics = set(current_topic_set)
            new_topics.update(candidate_reviews[idx]['topics'])
            added_topic_bonus = len(new_topics) - len(current_topic_set)
            local_score = candidate_reviews[idx]['trust'] + 0.2 * added_topic_bonus
            scored_next.append((local_score, idx))

        scored_next.sort(reverse=True)

        # only expand top few next moves from each state
        for _, idx in scored_next[:10]:
            new_selected = tuple(sorted(selected + (idx,)))
            new_remaining = tuple(r for r in remaining_list if r != idx)

            selected_topics = [candidate_reviews[s]['topics'] for s in new_selected]
            cov = _coverage(selected_topics) / topic_rule_count
            avg_trust = sum(candidate_reviews[s]['trust'] for s in new_selected) / len(new_selected)
            new_g = topic_weight * cov + trust_weight * avg_trust

            remaining_reviews_data = [candidate_reviews[r] for r in new_remaining]
            h = _heuristic(remaining_reviews_data) * trust_weight
            new_f = new_g + h

            heapq.heappush(heap, (-new_f, new_g, new_selected, new_remaining))

    if best_solution is None:
        # fallback: greedy top-k by trust
        sorted_idx = sorted(
            range(n),
            key=lambda i: candidate_reviews[i]['trust'],
            reverse=True
        )
        best_solution = tuple(sorted(sorted_idx[:k]))

    return [candidate_reviews[i] for i in best_solution]

# Step 4: Pretty-print the summary

def print_summary(selected_reviews, csp_report):
    print("\n" + "=" * 60)
    print("REVIEW SUMMARY")
    print("=" * 60)
    print(csp_report)
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


# Main

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
    }

    print("Applying CSP constraints...")
    filtered_df, csp_report = apply_constraints(df, constraints, text_col=TEXT_COL)

    candidates = [
        {
            'text': row[TEXT_COL],
            'topics': row['topics'],
            'sentiment_map': row['sentiment_map'],
            'trust': row['trust'],
            'trust_reasons': row['trust_reasons'],
        }
        for _, row in filtered_df.iterrows()
    ]

    print("Running A* search for optimal review subset...")
    selected = astar_search(
        candidates,
        k=5,
        beam_width=120,
        max_candidates=25,
        max_expansions=4000
    )

    print_summary(selected, csp_report)