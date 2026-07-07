import pytest


demo_adp = pytest.importorskip(
    "demo_adp",
    reason="demo_adp.py не восстанавливается в этой ветке",
)


def require_demo_api():
    assert hasattr(demo_adp, "DemoSettings")
    assert hasattr(demo_adp, "RunRecord")
    assert hasattr(demo_adp, "summarize_variant")
    assert hasattr(demo_adp, "overall_verdict")
    return (
        demo_adp.DemoSettings,
        demo_adp.RunRecord,
        demo_adp.summarize_variant,
        demo_adp.overall_verdict,
    )


def test_variant_summary_requires_each_trial_and_mean_thresholds():
    DemoSettings, RunRecord, summarize_variant, _ = require_demo_api()
    settings = DemoSettings(
        min_cosine_abs=0.90,
        min_mean_cosine_abs=0.92,
        max_mean_angle_deg=25.0,
    )
    good_records = [
        RunRecord("new", 0, 0.94, 19.9, 0.1, 1.0, 12, 15.0, True),
        RunRecord("new", 1, 0.96, 16.3, 0.1, 1.0, 12, 15.0, True),
    ]
    bad_records = [
        RunRecord("old", 0, 0.91, 24.5, 0.1, 1.0, 12, 15.0, True),
        RunRecord("old", 1, 0.89, 27.1, 0.1, 1.0, 12, 15.0, False),
    ]

    good_summary = summarize_variant("new", good_records, settings)
    bad_summary = summarize_variant("old", bad_records, settings)

    assert good_summary.passed
    assert not bad_summary.passed


def test_overall_verdict_requires_all_variants():
    DemoSettings, RunRecord, summarize_variant, overall_verdict = require_demo_api()
    settings = DemoSettings()
    passed_summary = summarize_variant(
        "new",
        [RunRecord("new", 0, 0.96, 16.3, 0.1, 1.0, 12, 15.0, True)],
        settings,
    )
    failed_summary = summarize_variant(
        "old",
        [RunRecord("old", 0, 0.10, 84.2, 0.1, 1.0, 12, 15.0, False)],
        settings,
    )

    assert not overall_verdict([passed_summary, failed_summary])
