#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Walk-Forward Backtesting Architecture for NCAA Basketball Prediction.

This script implements a day-by-day validation loop with lagged features,
rolling windows, and optional calibration/ROI tracking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

POSS_FTA_FACTOR = 0.475


@dataclass
class BacktestConfig:
    season_start: str
    season_end: str
    burn_in_seasons: list[int]
    target_season: int
    rolling_windows: tuple[int, ...] = (3, 5, 10)
    edge_threshold: float = 0.03
    output_dir: Path = Path("./backtest_output")


class DataIngestor:
    """Handles data acquisition and caching."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_games(self, season: int) -> pd.DataFrame:
        """Load games for a season via cache or cbbpy."""
        cache_path = self.cache_dir / f"games_{season}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        from importlib.util import find_spec

        if not find_spec("cbbpy"):
            raise RuntimeError("cbbpy is required to scrape games. Install cbbpy or provide cache.")

        from cbbpy.mens_scraper import get_games_season

        games = get_games_season(season)
        games["Date"] = pd.to_datetime(games["Date"])
        games.to_parquet(cache_path, index=False)
        return games

    def load_boxscore(self, game_id: str) -> pd.DataFrame:
        cache_path = self.cache_dir / f"boxscore_{game_id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        from importlib.util import find_spec

        if not find_spec("cbbpy"):
            raise RuntimeError("cbbpy is required to scrape boxscores. Install cbbpy or provide cache.")

        from cbbpy.mens_scraper import get_game_boxscore

        boxscore = get_game_boxscore(game_id)
        boxscore.to_parquet(cache_path, index=False)
        return boxscore


class FeatureCalculator:
    """Transforms raw boxscore data into Four Factors and tempo-free metrics."""

    @staticmethod
    def calc_possessions(fga: float, orb: float, tov: float, fta: float) -> float:
        return fga - orb + tov + (POSS_FTA_FACTOR * fta)

    def calc_four_factors(self, team_row: pd.Series, opp_row: pd.Series) -> dict:
        fg = team_row.get("FG", 0)
        fga = team_row.get("FGA", 0)
        fg3 = team_row.get("3P", team_row.get("3PM", 0))
        ft = team_row.get("FT", 0)
        fta = team_row.get("FTA", 0)
        orb = team_row.get("ORB", 0)
        drb = team_row.get("DRB", 0)
        tov = team_row.get("TOV", 0)
        pts = team_row.get("PTS", 0)

        opp_fg = opp_row.get("FG", 0)
        opp_fga = opp_row.get("FGA", 0)
        opp_fg3 = opp_row.get("3P", opp_row.get("3PM", 0))
        opp_fta = opp_row.get("FTA", 0)
        opp_orb = opp_row.get("ORB", 0)
        opp_drb = opp_row.get("DRB", 0)
        opp_tov = opp_row.get("TOV", 0)
        opp_pts = opp_row.get("PTS", 0)

        poss = self.calc_possessions(fga, orb, tov, fta)
        opp_poss = self.calc_possessions(opp_fga, opp_orb, opp_tov, opp_fta)
        pace = np.mean([poss, opp_poss]) if poss and opp_poss else poss or opp_poss

        efg = (fg + 0.5 * fg3) / fga if fga else 0.0
        tov_pct = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) else 0.0
        orb_pct = orb / (orb + opp_drb) if (orb + opp_drb) else 0.0
        ftr = fta / fga if fga else 0.0

        opp_efg = (opp_fg + 0.5 * opp_fg3) / opp_fga if opp_fga else 0.0
        opp_tov_pct = opp_tov / (opp_fga + 0.44 * opp_fta + opp_tov) if (opp_fga + 0.44 * opp_fta + opp_tov) else 0.0
        opp_orb_pct = opp_orb / (opp_orb + drb) if (opp_orb + drb) else 0.0
        opp_ftr = opp_fta / opp_fga if opp_fga else 0.0

        off_eff = (pts / poss) * 100 if poss else 0.0
        def_eff = (opp_pts / poss) * 100 if poss else 0.0

        return {
            "Team_eFG": efg,
            "Team_TOV": tov_pct,
            "Team_ORB": orb_pct,
            "Team_FTR": ftr,
            "Opp_eFG": opp_efg,
            "Opp_TOV": opp_tov_pct,
            "Opp_ORB": opp_orb_pct,
            "Opp_FTR": opp_ftr,
            "Off_Eff": off_eff,
            "Def_Eff": def_eff,
            "Possessions": pace,
        }

    def build_game_features(self, boxscore: pd.DataFrame) -> pd.DataFrame:
        team_rows = boxscore[boxscore["TeamType"] == "Team"].copy()
        if team_rows.empty:
            raise ValueError("Boxscore missing team-level rows.")

        output_rows = []
        grouped = team_rows.groupby("GameId")
        for game_id, group in grouped:
            if len(group) != 2:
                continue
            team_a, team_b = group.iloc[0], group.iloc[1]
            a_metrics = self.calc_four_factors(team_a, team_b)
            b_metrics = self.calc_four_factors(team_b, team_a)

            for team_row, metrics, opp_row in [
                (team_a, a_metrics, team_b),
                (team_b, b_metrics, team_a),
            ]:
                output_rows.append(
                    {
                        "GameId": game_id,
                        "Date": pd.to_datetime(team_row["Date"]),
                        "Season": team_row["Season"],
                        "Team": team_row["School"],
                        "Opponent": opp_row["School"],
                        "Loc": team_row.get("Location", "Neutral"),
                        "Result": team_row.get("Result", np.nan),
                        **metrics,
                    }
                )

        return pd.DataFrame(output_rows)

    def add_lagged_features(self, df: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
        df = df.sort_values(["Team", "Date"])
        for metric in ["Team_eFG", "Team_TOV", "Team_ORB", "Team_FTR", "Off_Eff", "Def_Eff"]:
            df[f"SeasonAvg_{metric}"] = (
                df.groupby("Team")[metric]
                .transform(lambda x: x.expanding().mean().shift(1))
                .astype(float)
            )
            for window in windows:
                df[f"Roll{window}_{metric}"] = (
                    df.groupby("Team")[metric]
                    .transform(lambda x: x.rolling(window=window).mean().shift(1))
                    .astype(float)
                )
        return df


class Backtester:
    """Walk-forward training and prediction loop."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.model = LogisticRegression(max_iter=200)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("Date")
        predictions = []
        train_df = df[df["Season"].isin(self.config.burn_in_seasons)].copy()

        start_date = pd.to_datetime(self.config.season_start)
        end_date = pd.to_datetime(self.config.season_end)

        target_df = df[(df["Season"] == self.config.target_season)].copy()
        target_df = target_df[(target_df["Date"] >= start_date) & (target_df["Date"] <= end_date)]

        unique_dates = sorted(target_df["Date"].unique())

        for current_date in unique_dates:
            day_games = target_df[target_df["Date"] == current_date].copy()

            features = [
                col
                for col in train_df.columns
                if col.startswith("SeasonAvg_") or col.startswith("Roll")
            ]

            train_subset = train_df.dropna(subset=features + ["Result"])
            if train_subset.empty:
                continue

            x_train = train_subset[features]
            y_train = (train_subset["Result"] == "W").astype(int)

            self.model.fit(x_train, y_train)

            day_games = day_games.dropna(subset=features)
            if day_games.empty:
                continue

            probs = self.model.predict_proba(day_games[features])[:, 1]
            for _, row in day_games.iterrows():
                predictions.append(
                    {
                        "GameId": row["GameId"],
                        "Date": row["Date"],
                        "Team": row["Team"],
                        "Opponent": row["Opponent"],
                        "Prob_Win": probs[day_games.index.get_loc(row.name)],
                        "Result": row["Result"],
                    }
                )

            # Update training set with results of current day
            train_df = pd.concat([train_df, day_games], ignore_index=True)

        return pd.DataFrame(predictions)


class Evaluator:
    """Compute validation metrics and calibration."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, predictions: pd.DataFrame) -> dict:
        predictions = predictions.dropna(subset=["Prob_Win", "Result"])
        y_true = (predictions["Result"] == "W").astype(int)
        y_prob = predictions["Prob_Win"].clip(0.001, 0.999)

        metrics = {
            "brier_score": brier_score_loss(y_true, y_prob),
            "log_loss": log_loss(y_true, y_prob),
        }

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        calib_df = pd.DataFrame({"prob_true": prob_true, "prob_pred": prob_pred})
        calib_path = self.output_dir / "calibration_curve.csv"
        calib_df.to_csv(calib_path, index=False)

        metrics_path = self.output_dir / "metrics.json"
        pd.Series(metrics).to_json(metrics_path)

        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward NCAA backtest")
    parser.add_argument("--season-start", default="2024-11-01")
    parser.add_argument("--season-end", default="2025-04-15")
    parser.add_argument("--target-season", type=int, default=2025)
    parser.add_argument("--burn-in", type=int, nargs="+", default=[2024])
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--output-dir", default="./backtest_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        season_start=args.season_start,
        season_end=args.season_end,
        burn_in_seasons=args.burn_in,
        target_season=args.target_season,
        output_dir=Path(args.output_dir),
    )

    ingestor = DataIngestor(cache_dir=Path(args.cache_dir))
    calculator = FeatureCalculator()

    all_games = []
    for season in config.burn_in_seasons + [config.target_season]:
        games = ingestor.load_games(season)
        all_games.append(games)

    games_df = pd.concat(all_games, ignore_index=True)
    game_ids = games_df["GameId"].unique().tolist()

    features_list = []
    for game_id in game_ids:
        boxscore = ingestor.load_boxscore(game_id)
        features_list.append(calculator.build_game_features(boxscore))

    features_df = pd.concat(features_list, ignore_index=True)
    features_df = calculator.add_lagged_features(features_df, config.rolling_windows)

    backtester = Backtester(config=config)
    predictions = backtester.run(features_df)

    evaluator = Evaluator(output_dir=config.output_dir)
    metrics = evaluator.evaluate(predictions)

    predictions_path = config.output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    print("Backtest complete.")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
