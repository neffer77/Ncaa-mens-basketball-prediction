#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NCAA BASKETBALL SYNDICATE ENGINE v4.0
-------------------------------------
Architecture: Elite Quantitative (Game Logs + Monte Carlo + Split Injury Model)
Author: Quant Researcher / Engineer
Dependencies: pandas, numpy, requests, scipy, beautifulsoup4

[!] CRITICAL UPGRADE: This version scrapes GAME LOGS to calculate true recency.
    It does not rely on static season averages for the prediction core.
"""

import sys
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.stats import norm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# User Agent to mimic browser traffic (prevents 403 Forbidden)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# Exponential Decay Factor for Recency (Alpha)
# Higher = More weight to recent games. 0.15 is a standard "Syndicate" setting.
RECENCY_ALPHA = 0.15

# Standard Home Court Advantage (Baseline)
# Adjusted dynamically if conference data suggests stronger HCA
BASELINE_HCA = 3.2

# Simulation Settings
SIM_ITERATIONS = 5000

# ==============================================================================
# CORE CLASSES
# ==============================================================================


class Player:
    def __init__(self, name, obpm, dbpm, mpg_pct):
        self.name = name
        self.obpm = float(obpm)
        self.dbpm = float(dbpm)
        self.mpg_pct = float(mpg_pct) / 100.0  # Convert 80% to 0.80

    @property
    def net_impact(self):
        # Value Over Replacement Player (approximate per 100 possessions)
        # Replacement level is roughly -2.0 BPM
        replacement_level = -2.0
        total_bpm = self.obpm + self.dbpm
        return (total_bpm - replacement_level) * self.mpg_pct


class TeamProfile:
    def __init__(self, name, year=2025):
        self.name = name
        self.year = year
        self.game_log = pd.DataFrame()
        self.roster = {}  # Map of Player Objects
        self.season_adj_o = 0.0
        self.season_adj_d = 0.0
        self.season_tempo = 0.0

        # Calculated Metrics
        self.recency_adj_o = 0.0
        self.recency_adj_d = 0.0
        self.recency_tempo = 0.0
        self.variance = 10.0  # Default std dev of performance

    def load_data(self):
        """Orchestrates data fetching."""
        print(f"[...] Building profile for {self.name}...")
        if self._fetch_torvik_profile():
            self._calculate_recency_metrics()
            return True
        return False

    def _fetch_torvik_profile(self):
        """
        Scrapes team page from BartTorvik to get Roster (BPM) and Game Log.
        URL format: https://barttorvik.com/team.php?team=Duke&year=2025
        """
        # Handle spaces in team names for URL
        formatted_name = self.name.replace(" ", "%20")
        url = f"https://barttorvik.com/team.php?team={formatted_name}&year={self.year}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"Error: Could not connect to data source for {self.name}")
                return False

            soup = BeautifulSoup(r.text, "html.parser")
            time.sleep(0.5)

            # --- 1. PARSE ROSTER (For Injury Calculation) ---
            tables = pd.read_html(r.text)
            player_df = None
            game_df = None

            for df in tables:
                # Identify Player Table: Looks for 'OBPM' and 'Min%'
                if "OBPM" in df.columns and "Min%" in df.columns:
                    player_df = df
                # Identify Game Log: Looks for 'Result' and 'Opponent'
                if "Opponent" in df.columns and "Result" in df.columns:
                    game_df = df

            if player_df is not None:
                # Clean columns
                player_df.columns = [c.replace(" ", "") for c in player_df.columns]
                for _, row in player_df.iterrows():
                    try:
                        p_name = str(row.get("Player", "Unknown"))
                        # Remove class year (e.g., "Cooper Flagg Fr")
                        p_name_clean = " ".join(p_name.split()[:-1])
                        obpm = row.get("OBPM", 0)
                        dbpm = row.get("DBPM", 0)
                        mpg = row.get("Min%", 0)
                        self.roster[p_name_clean.lower()] = Player(
                            p_name_clean, obpm, dbpm, mpg
                        )
                    except Exception:
                        continue

            if game_df is not None:
                self.game_log = game_df

            # Extract Season Baselines (Header usually contains AdjO/AdjD summaries)
            # For this script, we'll calculate baselines from the game log to ensure consistency
            return True

        except Exception as e:
            print(f"Scraping Error ({self.name}): {e}")
            return False

    def _calculate_recency_metrics(self):
        """
        Applies Exponential Weighted Moving Average (EWMA) to game logs.
        This captures 'Current Form' vs 'Season Average'.
        """
        if self.game_log.empty:
            print(f"Warning: No game log found for {self.name}. Using defaults.")
            self.recency_adj_o = 105.0
            self.recency_adj_d = 105.0
            self.recency_tempo = 70.0
            return

        try:
            # Drop rows that aren't games (headers often repeat)
            result_col = None
            for candidate in ["Result", "W/L"]:
                if candidate in self.game_log.columns:
                    result_col = candidate
                    break

            if result_col:
                gams = self.game_log[
                    self.game_log[result_col].astype(str).str.contains("W|L", na=False)
                ].copy()
            else:
                gams = self.game_log.copy()

            # Convert to numeric
            for col in ["AdjO", "AdjD", "Poss"]:
                if col in gams.columns:
                    gams[col] = pd.to_numeric(gams[col], errors="coerce")

            gams = gams.dropna(subset=["AdjO", "AdjD", "Poss"])

            if gams.empty:
                raise ValueError("No usable game log rows after cleaning.")

            # Apply EWMA (Exponential Weighted Moving Average)
            # We reverse the list so latest game is last for the calculation
            gams = gams.iloc[::-1]

            self.season_adj_o = gams["AdjO"].mean()
            self.season_adj_d = gams["AdjD"].mean()
            self.season_tempo = gams["Poss"].mean()

            # Recency Calculation
            self.recency_adj_o = (
                gams["AdjO"].ewm(alpha=RECENCY_ALPHA, adjust=False).mean().iloc[-1]
            )
            self.recency_adj_d = (
                gams["AdjD"].ewm(alpha=RECENCY_ALPHA, adjust=False).mean().iloc[-1]
            )
            self.recency_tempo = (
                gams["Poss"].ewm(alpha=RECENCY_ALPHA, adjust=False).mean().iloc[-1]
            )

            # Calculate Variance (Standard Deviation of Game Scores)
            # Used for Monte Carlo
            # Approx game margin std dev derived from efficiency variance
            self.variance = gams["AdjO"].std() + gams["AdjD"].std()

        except Exception as e:
            print(f"Math Error ({self.name}): {e}")
            # Fallback
            self.recency_adj_o = 105.0
            self.recency_adj_d = 105.0
            self.recency_tempo = 70.0

    def get_injury_impact(self, injured_names):
        """
        Returns separate Offense and Defense adjustments.
        """
        adj_o_drop = 0.0
        adj_d_rise = 0.0  # Defense getting worse = Rating goes UP (points allowed)

        report = []

        for name in injured_names:
            name_clean = name.lower().strip()
            # Fuzzy match
            found = None
            for r_name in self.roster:
                if name_clean in r_name:
                    found = self.roster[r_name]
                    break

            if found:
                # Calculate loss vs replacement
                # Offense: Lose OBPM, gain Replacement (-1.0 approx for offense)
                # Defense: Lose DBPM, gain Replacement (-1.0 approx for defense)

                # If player is +4.0 OBPM, impact is 4.0 - (-1.0) = 5.0 lost * %Min
                lost_o = (found.obpm - (-1.0)) * found.mpg_pct
                lost_d = (found.dbpm - (-1.0)) * found.mpg_pct

                adj_o_drop += max(0, lost_o)
                adj_d_rise += max(0, lost_d)  # Loss of good defender adds points to AdjD

                report.append(f"{found.name} (OBPM:{found.obpm}, DBPM:{found.dbpm})")
            else:
                print(f"  [!] Warning: Player '{name}' not found in roster.")

        return adj_o_drop, adj_d_rise, report


# ==============================================================================
# MONTE CARLO ENGINE
# ==============================================================================


class SyndicateSim:
    def __init__(self, home, away, neutral=False):
        self.home = home
        self.away = away
        self.neutral = neutral

    def run(self, h_injuries, a_injuries):
        # 1. Apply Injuries
        h_o_drop, h_d_rise, h_report = self.home.get_injury_impact(h_injuries)
        a_o_drop, a_d_rise, a_report = self.away.get_injury_impact(a_injuries)

        # 2. Adjust Ratings (Recency + Injury)
        h_final_o = self.home.recency_adj_o - h_o_drop
        h_final_d = self.home.recency_adj_d + h_d_rise

        a_final_o = self.away.recency_adj_o - a_o_drop
        a_final_d = self.away.recency_adj_d + a_d_rise

        # 3. Calculate Pace
        pace = (self.home.recency_tempo + self.away.recency_tempo) / 2.0

        # 4. HCA
        hca = 0 if self.neutral else BASELINE_HCA

        # 5. Expected Margin (Efficiency Differential)
        # KenPom/Torvik Approx: (HomeAdjO - AwayAdjD) + (HomeAdjD - AwayAdjO)??
        # Simplified: (HomeNet - AwayNet) adjusted for Pace + HCA

        # Expected Points Per 100
        # Home vs Away Defense
        h_ppp = (h_final_o + a_final_d) / 2.0
        a_ppp = (a_final_o + h_final_d) / 2.0

        # 6. Monte Carlo Loop
        margins = []
        total_pts = []

        # Volatility factors
        h_std = self.home.variance * (pace / 100.0)
        a_std = self.away.variance * (pace / 100.0)

        for _ in range(SIM_ITERATIONS):
            # Simulate Offense Performance based on volatility
            h_perf = np.random.normal(h_ppp, h_std / 2)  # div 2 smoothing
            a_perf = np.random.normal(a_ppp, a_std / 2)

            # Convert to Game Score
            h_score_sim = (h_perf * (pace / 100.0)) + (hca / 2)
            a_score_sim = (a_perf * (pace / 100.0)) - (hca / 2)

            margins.append(a_score_sim - h_score_sim)  # Away - Home (neg = home win)
            total_pts.append(h_score_sim + a_score_sim)

        # 7. Analysis
        avg_margin = np.mean(margins)  # Away - Home
        avg_total = np.mean(total_pts)
        win_prob_home = np.mean([m < 0 for m in margins])

        return {
            "spread_fair": avg_margin,  # If -5, Home is favored by 5
            "total_fair": avg_total,
            "win_prob": win_prob_home,
            "home_stats": (h_final_o, h_final_d),
            "away_stats": (a_final_o, a_final_d),
            "pace": pace,
            "injuries": {"home": h_report, "away": a_report},
        }


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================


def get_kelly_stake(prob_win, odds_american, bankroll_pct=0.05):
    """Calculates Kelly Criterion stake size."""
    if odds_american < 0:
        decimal_odds = (100 / abs(odds_american)) + 1
    else:
        decimal_odds = (odds_american / 100) + 1

    b = decimal_odds - 1
    q = 1 - prob_win
    f = (b * prob_win - q) / b

    # Fractional Kelly (Half-Kelly is safer for sports)
    safe_f = f * 0.5

    if safe_f <= 0:
        return 0.0
    return min(safe_f, bankroll_pct) * 100  # Return as % of bankroll (capped)


def main():
    print("\n" + "=" * 70)
    print("SYNDICATE V4.0 | ELITE NCAA PREDICTIVE ENGINE")
    print("Features: Game Log Recency | Split Injury Logic | Monte Carlo Sim")
    print("=" * 70)

    # User Input
    home_input = input("Enter Home Team (e.g. Duke): ").strip()
    away_input = input("Enter Away Team (e.g. North Carolina): ").strip()
    is_neutral = input("Neutral Site? (y/n): ").lower() == "y"

    # Build Profiles
    home_team = TeamProfile(home_input)
    away_team = TeamProfile(away_input)

    if not home_team.load_data() or not away_team.load_data():
        print("System Halted: Data fetch failed.")
        sys.exit()

    # Check for Injuries
    print(f"\n[?] Roster Check for {home_team.name}:")
    h_inj = input("    List OUT players (comma sep) or hit ENTER: ").split(",")
    h_inj = [x.strip() for x in h_inj if x.strip()]

    print(f"[?] Roster Check for {away_team.name}:")
    a_inj = input("    List OUT players (comma sep) or hit ENTER: ").split(",")
    a_inj = [x.strip() for x in a_inj if x.strip()]

    # Run Simulation
    print("\n")
    sim = SyndicateSim(home_team, away_team, is_neutral)
    res = sim.run(h_inj, a_inj)

    # --- REPORT GENERATION ---
    print("\n" + "=" * 70)
    print(f"OFFICIAL SYNDICATE REPORT: {away_team.name} @ {home_team.name}")
    print("=" * 70)

    # Prediction
    fair_spread = res["spread_fair"]
    favored = home_team.name if fair_spread < 0 else away_team.name
    line_display = abs(fair_spread)

    print(
        f"\nPREDICTED SCORE: {home_team.name} {(res['total_fair'] / 2 - fair_spread / 2):.1f} - "
        f"{away_team.name} {(res['total_fair'] / 2 + fair_spread / 2):.1f}"
    )
    print(f"FAIR LINE:       {favored} -{line_display:.2f}")
    print(f"FAIR TOTAL:      {res['total_fair']:.2f}")
    print(f"WIN PROBABILITY: {home_team.name} {res['win_prob'] * 100:.1f}%")

    # Recency Delta
    print("\n--- RECENCY & FORM ---")
    print(
        f"{home_team.name}: Season AdjO {home_team.season_adj_o:.1f} -> "
        f"Current Form {home_team.recency_adj_o:.1f}"
    )
    print(
        f"{away_team.name}: Season AdjO {away_team.season_adj_o:.1f} -> "
        f"Current Form {away_team.recency_adj_o:.1f}"
    )

    # Injury Report
    if res["injuries"]["home"] or res["injuries"]["away"]:
        print("\n--- INJURY ADJUSTMENTS APPLIED ---")
        for p in res["injuries"]["home"]:
            print(f"  {home_team.name}: {p}")
        for p in res["injuries"]["away"]:
            print(f"  {away_team.name}: {p}")

    # --- BETTING VALUE CALCULATOR ---
    print("\n" + "-" * 70)
    print("MARKET ANALYSIS (KELLY CRITERION)")
    print("-" * 70)

    try:
        market_line_str = input(f"Enter Current Market Spread for {favored} (e.g. -4.5): ")
        market_odds_str = input("Enter Odds (e.g. -110): ")

        market_line = float(market_line_str)
        market_odds = float(market_odds_str)

        # Calculate Edge
        # If Model says -7.5 and Market is -4.5, we have 3 points of value
        model_margin = -1 * fair_spread if favored == home_team.name else fair_spread
        market_margin = abs(market_line)  # Assuming user input negative for fav

        edge = model_margin - market_margin

        print(f"\nModel Margin: {model_margin:.2f} | Market Margin: {market_margin:.2f}")
        print(f"Discrepancy: {edge:.2f} points")

        if edge > 1.5:
            # Simple Kelly Logic based on Win Prob of the bet
            # We need to map the point edge to a probability advantage.
            # Rule of thumb: 1 point of edge ~ 3-4% win prob shift near spread of 0

            print(f"VERDICT: *** VALUE DETECTED on {favored} ***")

            # Approximate probability of covering based on edge (Simple Heuristic)
            # Standard deviation of NCAA game is ~11.
            # Z-score of edge
            z = edge / 11.0
            prob_cover = norm.cdf(z * 0.7) + 0.5  # Damping factor
            prob_cover = min(prob_cover, 0.99)  # Cap

            rec_bet = get_kelly_stake(prob_cover, market_odds)
            print(f"Est. Cover Probability: {prob_cover * 100:.1f}%")
            print(f"Recommended Kelly Wager: {rec_bet:.2f}% of Bankroll")

        elif edge < -1.5:
            print(
                f"VERDICT: VALUE ON UNDERDOG ({home_team.name if favored != home_team.name else away_team.name})"
            )
        else:
            print("VERDICT: NO PLAY (Market is Efficient)")

    except ValueError:
        print("Invalid input, skipping market analysis.")

    print("=" * 70)


if __name__ == "__main__":
    main()
