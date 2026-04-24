"""
test_csp.py — Standalone tests for the CSP solver.
Run from src/: python test_csp.py
"""
from csp_solver import CSPSolver


def make_review(text, topics, trust, sentiment_map=None):
    return {
        'text': text,
        'topics': topics,
        'trust': trust,
        'sentiment_map': sentiment_map or {t: 'Positive' for t in topics},
        'trust_reasons': [],
    }


def test_basic_solution():
    """CSP should find k=2 reviews from a clean pool."""
    candidates = [
        make_review("Great fabric, very soft", ['Quality'], 0.9),
        make_review("Shipping was late again", ['Logistics'], 0.8),
        make_review("Fits perfectly love it", ['Fit'], 0.85),
    ]
    constraints = {'min_trust': 0.7, 'min_words': 3, 'required_topic': None, 'sentiment_filter': None, 'coverage_goal': True}
    solver = CSPSolver(candidates, k=2, constraints=constraints)
    solution, stats, log = solver.solve()
    assert solution is not None, "Expected a solution"
    assert len(solution) == 2, f"Expected 2 reviews, got {len(solution)}"
    print(f"PASS test_basic_solution — nodes expanded: {stats['nodes_expanded']}, backtracks: {stats['backtracks']}")


def test_trust_constraint_filters():
    """Reviews below min_trust should be excluded via unary constraint."""
    candidates = [
        make_review("Great fabric", ['Quality'], 0.9),
        make_review("Bad", ['General'], 0.2),   # should be filtered out
        make_review("Nice fit", ['Fit'], 0.8),
    ]
    constraints = {'min_trust': 0.6, 'min_words': 1, 'required_topic': None, 'sentiment_filter': None, 'coverage_goal': False}
    solver = CSPSolver(candidates, k=2, constraints=constraints)
    assert len(solver.initial_domain) == 2, f"Expected 2 after trust filter, got {len(solver.initial_domain)}"
    solution, stats, log = solver.solve()
    assert solution is not None
    assert all(r['trust'] >= 0.6 for r in solution), "Trust constraint violated"
    print(f"PASS test_trust_constraint_filters — domain size: {len(solver.initial_domain)}")


def test_required_topic_satisfied():
    """CSP must include at least one review covering required_topic."""
    candidates = [
        make_review("Great fabric quality durable material nice", ['Quality'], 0.9),
        make_review("Shipping was delayed and late arrived wrong", ['Logistics'], 0.8),
        make_review("Fits well size is perfect large", ['Fit'], 0.85),
    ]
    constraints = {'min_trust': 0.7, 'min_words': 3, 'required_topic': 'Logistics', 'sentiment_filter': None, 'coverage_goal': False}
    solver = CSPSolver(candidates, k=2, constraints=constraints)
    solution, stats, log = solver.solve()
    assert solution is not None, "Expected solution covering Logistics"
    covered_topics = set()
    for r in solution:
        covered_topics.update(r['topics'])
    assert 'Logistics' in covered_topics, f"Logistics not covered — got {covered_topics}"
    print(f"PASS test_required_topic_satisfied — topics covered: {covered_topics}")


def test_impossible_constraint_returns_none():
    """If no review meets the required_topic, CSP should return None."""
    candidates = [
        make_review("Great fabric quality durable", ['Quality'], 0.9),
        make_review("Fits well size perfect", ['Fit'], 0.85),
    ]
    constraints = {'min_trust': 0.7, 'min_words': 3, 'required_topic': 'Logistics', 'sentiment_filter': None, 'coverage_goal': False}
    solver = CSPSolver(candidates, k=2, constraints=constraints)
    solution, stats, log = solver.solve()
    assert solution is None, f"Expected None for impossible constraint, got {solution}"
    print(f"PASS test_impossible_constraint_returns_none — backtracks: {stats['backtracks']}")


def test_no_duplicates_in_solution():
    """The same review must not appear in two slots."""
    candidates = [
        make_review("Only review that passes trust", ['Quality'], 0.95),
        make_review("Second review just barely", ['Fit'], 0.80),
    ]
    constraints = {'min_trust': 0.7, 'min_words': 3, 'required_topic': None, 'sentiment_filter': None, 'coverage_goal': False}
    solver = CSPSolver(candidates, k=2, constraints=constraints)
    solution, stats, log = solver.solve()
    assert solution is not None
    texts = [r['text'] for r in solution]
    assert len(texts) == len(set(texts)), f"Duplicate reviews in solution: {texts}"
    print(f"PASS test_no_duplicates_in_solution")


def test_forward_checking_prunes():
    """Forward checking should prune duplicate from future domains."""
    candidates = [
        make_review("Review A quality fabric durable", ['Quality'], 0.9),
        make_review("Review B logistics late shipping", ['Logistics'], 0.8),
        make_review("Review C fit size large", ['Fit'], 0.85),
    ]
    constraints = {'min_trust': 0.7, 'min_words': 3, 'required_topic': None, 'sentiment_filter': None, 'coverage_goal': False}
    solver = CSPSolver(candidates, k=2, constraints=constraints)
    solution, stats, log = solver.solve()
    assert solution is not None
    assert stats['forward_prunes'] > 0, "Expected at least one forward prune (duplicate removal)"
    print(f"PASS test_forward_checking_prunes — forward prunes: {stats['forward_prunes']}")


if __name__ == '__main__':
    print("Running CSP solver tests...\n")
    test_basic_solution()
    test_trust_constraint_filters()
    test_required_topic_satisfied()
    test_impossible_constraint_returns_none()
    test_no_duplicates_in_solution()
    test_forward_checking_prunes()
    print("\nAll tests passed.")