from atlas_rtc.detectors.heuristic import HeuristicDriftDetector


def test_heuristic_detector_higher_for_invalid_mass():
    detector = HeuristicDriftDetector()
    low = detector.score({"invalid_mass": 0.1, "entropy": 0.2, "opened": 1.0, "step_index": 3.0})
    high = detector.score({"invalid_mass": 0.8, "entropy": 0.2, "opened": 1.0, "step_index": 3.0})
    assert high > low
