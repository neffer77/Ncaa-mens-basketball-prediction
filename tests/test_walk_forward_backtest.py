from pathlib import Path

import numpy as np
import pandas as pd

from walk_forward_backtest import BacktestConfig, Backtester, Evaluator, FeatureCalculator


def test_calc_possessions():
    calc = FeatureCalculator()
    assert calc.calc_possessions(60, 10, 12, 20) == 60 - 10 + 12 + (0.475 * 20)


def test_calc_four_factors():
    calc = FeatureCalculator()
    team = pd.Series(
        {
            "FG": 30,
            "FGA": 60,
            "3P": 6,
            "FT": 12,
            "FTA": 18,
            "ORB": 8,
            "DRB": 22,
            "TOV": 10,
            "PTS": 78,
        }
    )
    opp = pd.Series(
        {
            "FG": 28,
            "FGA": 58,
            "3P": 4,
            "FTA": 16,
            "ORB": 7,
            "DRB": 20,
            "TOV": 12,
            "PTS": 72,
        }
    )

    metrics = calc.calc_four_factors(team, opp)

    expected_efg = (30 + 0.5 * 6) / 60
    expected_tov = 10 / (60 + 0.44 * 18 + 10)
    expected_orb = 8 / (8 + 20)
    expected_ftr = 18 / 60

    assert np.isclose(metrics["Team_eFG"], expected_efg)
    assert np.isclose(metrics["Team_TOV"], expected_tov)
    assert np.isclose(metrics["Team_ORB"], expected_orb)
    assert np.isclose(metrics["Team_FTR"], expected_ftr)
    assert metrics["Possessions"] > 0


def test_add_lagged_features_creates_shifted_metrics():
    calc = FeatureCalculator()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-11-01", "2024-11-05", "2024-11-10"]),
            "Team": ["A", "A", "A"],
            "Team_eFG": [0.5, 0.55, 0.6],
            "Team_TOV": [0.15, 0.14, 0.13],
            "Team_ORB": [0.3, 0.32, 0.31],
            "Team_FTR": [0.2, 0.22, 0.21],
            "Off_Eff": [110, 115, 112],
            "Def_Eff": [100, 98, 102],
        }
    )

    out = calc.add_lagged_features(df, windows=(2,))

    assert np.isnan(out.loc[0, "SeasonAvg_Team_eFG"])
    assert np.isclose(out.loc[1, "SeasonAvg_Team_eFG"], 0.5)
    assert np.isclose(out.loc[2, "SeasonAvg_Team_eFG"], (0.5 + 0.55) / 2)
    assert np.isnan(out.loc[0, "Roll2_Team_eFG"])
    assert np.isnan(out.loc[1, "Roll2_Team_eFG"])
    assert np.isclose(out.loc[2, "Roll2_Team_eFG"], (0.5 + 0.55) / 2)


def test_backtester_run_generates_predictions():
    config = BacktestConfig(
        season_start="2025-01-01",
        season_end="2025-01-02",
        burn_in_seasons=[2024],
        target_season=2025,
    )
    backtester = Backtester(config)

    df = pd.DataFrame(
        {
            "GameId": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "Date": pd.to_datetime(
                [
                    "2024-12-01",
                    "2024-12-05",
                    "2024-12-10",
                    "2024-12-15",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "Season": [2024, 2024, 2024, 2024, 2025, 2025],
            "Team": ["A", "B", "C", "D", "A", "B"],
            "Opponent": ["B", "A", "D", "C", "C", "D"],
            "Result": ["W", "L", "W", "L", "W", "L"],
            "SeasonAvg_Team_eFG": [0.5, 0.48, 0.52, 0.49, 0.51, 0.47],
            "Roll3_Team_eFG": [0.5, 0.48, 0.52, 0.49, 0.51, 0.47],
        }
    )

    predictions = backtester.run(df)

    assert len(predictions) == 2
    assert predictions["Prob_Win"].between(0, 1).all()


def test_evaluator_writes_outputs(tmp_path: Path):
    evaluator = Evaluator(output_dir=tmp_path)
    predictions = pd.DataFrame(
        {
            "Prob_Win": [0.6, 0.4],
            "Result": ["W", "L"],
        }
    )

    metrics = evaluator.evaluate(predictions)

    assert "brier_score" in metrics
    assert (tmp_path / "calibration_curve.csv").exists()
    assert (tmp_path / "metrics.json").exists()
