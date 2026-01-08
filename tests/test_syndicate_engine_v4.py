import numpy as np
import pandas as pd

from syndicate_engine_v4 import (
    BASELINE_HCA,
    RECENCY_ALPHA,
    Player,
    SyndicateSim,
    TeamProfile,
    get_kelly_stake,
)


def test_calculate_recency_metrics_uses_game_log():
    team = TeamProfile("Test")
    team.game_log = pd.DataFrame(
        {
            "Result": ["W", "L", "W"],
            "AdjO": [110.0, 100.0, 120.0],
            "AdjD": [95.0, 105.0, 90.0],
            "Poss": [70.0, 68.0, 72.0],
        }
    )

    team._calculate_recency_metrics()

    gams = team.game_log.iloc[::-1].reset_index(drop=True)
    expected_adj_o = gams["AdjO"].ewm(alpha=RECENCY_ALPHA, adjust=False).mean().iloc[-1]
    expected_adj_d = gams["AdjD"].ewm(alpha=RECENCY_ALPHA, adjust=False).mean().iloc[-1]
    expected_poss = gams["Poss"].ewm(alpha=RECENCY_ALPHA, adjust=False).mean().iloc[-1]

    assert team.season_adj_o == gams["AdjO"].mean()
    assert team.season_adj_d == gams["AdjD"].mean()
    assert team.season_tempo == gams["Poss"].mean()
    assert team.recency_adj_o == expected_adj_o
    assert team.recency_adj_d == expected_adj_d
    assert team.recency_tempo == expected_poss


def test_calculate_recency_metrics_defaults_when_empty():
    team = TeamProfile("Empty")
    team._calculate_recency_metrics()

    assert team.recency_adj_o == 105.0
    assert team.recency_adj_d == 105.0
    assert team.recency_tempo == 70.0


def test_get_injury_impact_matches_roster():
    team = TeamProfile("Roster")
    team.roster = {
        "alice guard": Player("Alice Guard", 4.0, 1.0, 80),
        "bob forward": Player("Bob Forward", -1.0, -2.0, 50),
    }

    adj_o_drop, adj_d_rise, report = team.get_injury_impact(["alice", "missing"])

    expected_o = (4.0 - (-1.0)) * 0.8
    expected_d = (1.0 - (-1.0)) * 0.8

    assert np.isclose(adj_o_drop, expected_o)
    assert np.isclose(adj_d_rise, expected_d)
    assert "Alice Guard" in report[0]


def test_syndicate_sim_run_is_deterministic_with_fixed_random(monkeypatch):
    home = TeamProfile("Home")
    away = TeamProfile("Away")
    home.recency_adj_o = 110.0
    home.recency_adj_d = 100.0
    home.recency_tempo = 70.0
    home.variance = 8.0

    away.recency_adj_o = 105.0
    away.recency_adj_d = 102.0
    away.recency_tempo = 70.0
    away.variance = 8.0

    def fixed_normal(mean, _std):
        return mean

    monkeypatch.setattr(np.random, "normal", fixed_normal)

    sim = SyndicateSim(home, away, neutral=False)
    res = sim.run([], [])

    h_ppp = (home.recency_adj_o + away.recency_adj_d) / 2.0
    a_ppp = (away.recency_adj_o + home.recency_adj_d) / 2.0
    pace = 70.0

    h_score = (h_ppp * (pace / 100.0)) + (BASELINE_HCA / 2)
    a_score = (a_ppp * (pace / 100.0)) - (BASELINE_HCA / 2)

    expected_margin = a_score - h_score
    expected_total = a_score + h_score

    assert np.isclose(res["spread_fair"], expected_margin)
    assert np.isclose(res["total_fair"], expected_total)
    assert res["win_prob"] == (1.0 if expected_margin < 0 else 0.0)


def test_get_kelly_stake_zero_when_negative_edge():
    stake = get_kelly_stake(prob_win=0.4, odds_american=-110)
    assert stake == 0.0
