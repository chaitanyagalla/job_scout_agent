from __future__ import annotations

from job_scout.evaluation import default_fixture_path
from job_scout.evaluation import load_evaluation_suite
from job_scout.evaluation import run_evaluation_suite


def test_load_evaluation_suite_reads_expected_fixture_groups():
    suite = load_evaluation_suite(default_fixture_path())

    assert len(suite.scoring_cases) >= 2
    assert len(suite.ranking_cases) >= 1
    assert len(suite.experience_filter_cases) >= 1


def test_evaluation_suite_passes_all_cases():
    summary = run_evaluation_suite(default_fixture_path())

    assert summary["failed"] == 0
    assert summary["passed"] == summary["total"]
