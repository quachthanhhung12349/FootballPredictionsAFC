import os
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from collections import deque, defaultdict

# --- Configuration & Setup ---
# Use absolute path for .env to ensure it's loaded correctly
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import kaggle AFTER loading environment variables
from kaggle.api.kaggle_api_extended import KaggleApi
import xgboost as xgb
import joblib

def setup_kaggle():
    """Authenticate with Kaggle API."""
    api = KaggleApi()
    api.authenticate()
    return api

def download_data(api, dataset_path='../../data'):
    """Download the football results dataset from Kaggle."""
    api.dataset_download_files(
        'martj42/international-football-results-from-1872-to-2017',
        path=dataset_path,
        unzip=True,
        force=True
    )
    print(f"Dataset downloaded to: {dataset_path}")

# --- Helper Functions ---

def team_name_to_slug(team_name: str) -> str:
    """Convert team names to slug format used by the country-year scraper output."""
    alias = {
        "China PR": "China",
        "Saudi Arabia": "Saudi_Arabia",
        "United Arab Emirates": "United_Arab_Emirates",
        "South Korea": "South_Korea",
        "North Korea": "North_Korea",
        "Hong Kong": "Hong_Kong",
        "Northern Mariana Islands": "Northern_Mariana_Islands",
        "Timor-Leste": "East_Timor",
    }
    if team_name in alias:
        return alias[team_name]
    return str(team_name).replace("-", "_").replace(" ", "_")

def parse_height_to_m(value):
    """Convert height like '1.77m' or '177 cm' to meters float."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan

    m_match = re.search(r"(\d+(?:\.\d+)?)\s*m", text)
    if m_match:
        return float(m_match.group(1))

    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", text)
    if cm_match:
        return float(cm_match.group(1)) / 100.0

    try:
        numeric = float(text)
        return numeric if numeric < 3 else numeric / 100.0
    except ValueError:
        return np.nan

def tournament_weight(tournament):
    """Assign weights based on tournament importance."""
    asian_world_quals = ['AFC Asian Cup qualification', 'FIFA World Cup qualification']
    asian_world_finals = ['AFC Asian Cup', 'FIFA World Cup']
    challenge_cup = ['AFC Challenge Cup', 'AFC Challenge Cup qualification']
    regional_cups = ['WAFF Championship', 'AFF Championship', 'Gulf Cup', 'Arab Cup', 'SAFF Cup', 'EAFF Championship', 'East Asian Games', 'ASEAN Championship']
    regional_cups_quals = ['AFF Championship qualification', 'Arab Cup qualification', 'ASEAN Championship qualification', 'EAFF Championship qualification']
    
    if tournament in asian_world_quals:
        return 20
    elif tournament in asian_world_finals:
        return 25
    elif tournament in challenge_cup:
        return 15
    elif tournament in regional_cups:
        return 15
    elif tournament in regional_cups_quals:
        return 10
    elif tournament == 'Forfeited':
        return 1
    else:
        return 5

# --- Glicko-1 System ---

def update_glicko(r1, rd1, r2, rd2, home_score, away_score, tournament):
    """Standard Glicko-1 update with modifications for goals and tournament weights."""
    q = math.log(10) / 400
    g_rd2 = 1 / math.sqrt(1 + 3 * (q**2) * (rd2**2) / (math.pi**2))
    expected = 1 / (1 + 10**(-g_rd2 * (r1 - r2) / 400))
    
    if home_score > away_score: outcome = 1
    elif home_score < away_score: outcome = 0
    else: outcome = 0.5
    
    n = abs(home_score - away_score)
    if n <= 1: g_index = 1
    elif n == 2: g_index = 1.5
    elif n == 3: g_index = 2
    elif n == 4: g_index = 2.25
    else: g_index = 1.75 + n / 8
        
    k = tournament_weight(tournament)
    d_squared = 1 / (q**2 * g_rd2**2 * expected * (1 - expected))
    rating_change = (q / (1/rd1**2 + 1/d_squared)) * g_rd2 * (outcome - expected)
    new_r = r1 + (rating_change * g_index * (k/20))
    new_rd = math.sqrt(1 / (1/rd1**2 + 1/d_squared))
    
    return new_r, new_rd

def get_pre_match_glicko(df, teams_list):
    """Compute pre-match Glicko ratings for all teams in the dataframe."""
    current_ratings = {team: (1500.0, 350.0) for team in teams_list}
    processed_data = []
    df_sorted = df.sort_values('date').copy()
    
    for _, row in df_sorted.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_r, home_rd = current_ratings[home_team]
        away_r, away_rd = current_ratings[away_team]
        
        match_info = row.to_dict()
        match_info['home_glicko_rating'] = home_r
        match_info['home_glicko_rd'] = home_rd
        match_info['away_glicko_rating'] = away_r
        match_info['away_glicko_rd'] = away_rd
        match_info['glicko_rating_diff'] = home_r - away_r
        
        new_rh, new_rdh = update_glicko(home_r, home_rd, away_r, away_rd, 
                                        row['home_score'], row['away_score'], 
                                        row['tournament'])
        new_ra, new_rda = update_glicko(away_r, away_rd, home_r, home_rd, 
                                        row['away_score'], row['home_score'], 
                                        row['tournament'])
        
        match_info['home_glicko_rating_post'] = new_rh
        match_info['away_glicko_rating_post'] = new_ra
        processed_data.append(match_info)
        
        current_ratings[home_team] = (new_rh, new_rdh)
        current_ratings[away_team] = (new_ra, new_rda)
        
    return pd.DataFrame(processed_data)

# --- Feature Engineering Functions ---

def calculate_streak(history):
    if not history: return 0
    recent = list(history)
    last = recent[-1]
    if last['W'] == 1:
        streak = 0
        for match in reversed(recent):
            if match['L'] == 0: streak += 1
            else: break
        return streak
    streak = 0
    for match in reversed(recent):
        if match['W'] == 0: streak += 1
        else: break
    return -streak

def calculate_form(history):
    games_played = len(history)
    if not history: return 0, 0, 0, 0, 0, 1500.0, 0, 0, 0, 0, 0
    wins = sum(x['W'] for x in history)
    draws = sum(x['D'] for x in history)
    losses = sum(x['L'] for x in history)
    gf = sum(x['GF'] for x in history)
    ga = sum(x['GA'] for x in history)
    avg_opp_rating = sum(x['opp_post_rating'] for x in history) / games_played
    
    win_opp_ratings = [x['opp_post_rating'] for x in history if x['W'] == 1]
    draw_opp_ratings = [x['opp_post_rating'] for x in history if x['D'] == 1]
    loss_opp_ratings = [x['opp_post_rating'] for x in history if x['L'] == 1]

    return (
        wins / games_played,
        draws / games_played,
        losses / games_played,
        gf / games_played,
        ga / games_played,
        avg_opp_rating,
        games_played,
        sum(win_opp_ratings) / len(win_opp_ratings) if win_opp_ratings else 0,
        sum(draw_opp_ratings) / len(draw_opp_ratings) if draw_opp_ratings else 0,
        sum(loss_opp_ratings) / len(loss_opp_ratings) if loss_opp_ratings else 0,
        calculate_streak(history)
    )

def add_form_features(glicko_df, teams_list):
    team_history = {team: deque(maxlen=10) for team in teams_list}
    processed_data = []
    
    for _, row in glicko_df.iterrows():
        home_team, away_team = row['home_team'], row['away_team']
        h_stats = calculate_form(team_history[home_team])
        a_stats = calculate_form(team_history[away_team])
        
        match_info = row.to_dict()
        pref_h = 'home_form_'
        pref_a = 'away_form_'
        cols = ['W', 'D', 'L', 'GF', 'GA', 'avg_opp_rating', 'games_max10', 'avg_opp_rating_W', 'avg_opp_rating_D', 'avg_opp_rating_L', 'streak']
        
        for i, col in enumerate(cols):
            match_info[pref_h + col] = h_stats[i]
            match_info[pref_a + col] = a_stats[i]
            
        processed_data.append(match_info)
        
        h_s, a_s = row['home_score'], row['away_score']
        team_history[home_team].append({'W': 1 if h_s > a_s else 0, 'D': 1 if h_s == a_s else 0, 'L': 1 if h_s < a_s else 0, 'GF': h_s, 'GA': a_s, 'opp_post_rating': row['away_glicko_rating_post']})
        team_history[away_team].append({'W': 1 if a_s > h_s else 0, 'D': 1 if a_s == h_s else 0, 'L': 1 if a_s < h_s else 0, 'GF': a_s, 'GA': h_s, 'opp_post_rating': row['home_glicko_rating_post']})
        
    return pd.DataFrame(processed_data)

def add_h2h_features(df):
    h2h_history = defaultdict(lambda: deque(maxlen=5))
    processed_data = []
    
    for _, row in df.iterrows():
        home_team, away_team = row['home_team'], row['away_team']
        pair_key = tuple(sorted([home_team, away_team]))
        history = h2h_history[pair_key]
        
        h2h_games = len(history)
        h2h_stats = {'h_win': 0, 'a_win': 0, 'draw': 0, 'h_goals': 0, 'a_goals': 0}
        
        for past in history:
            if past['winner'] == home_team: h2h_stats['h_win'] += 1
            elif past['winner'] == away_team: h2h_stats['a_win'] += 1
            else: h2h_stats['draw'] += 1
            
            if past['home'] == home_team:
                h2h_stats['h_goals'] += past['h_score']
                h2h_stats['a_goals'] += past['a_score']
            else:
                h2h_stats['h_goals'] += past['a_score']
                h2h_stats['a_goals'] += past['h_score']
                
        match_info = row.to_dict()
        match_info['H2H_home_team_win'] = h2h_stats['h_win'] / h2h_games if h2h_games > 0 else 0
        match_info['H2H_draw'] = h2h_stats['draw'] / h2h_games if h2h_games > 0 else 0
        match_info['H2H_away_team_win'] = h2h_stats['a_win'] / h2h_games if h2h_games > 0 else 0
        match_info['H2H_games_max5'] = h2h_games
        match_info['H2H_home_team_score_avg'] = h2h_stats['h_goals'] / h2h_games if h2h_games > 0 else 0
        match_info['H2H_away_team_score_avg'] = h2h_stats['a_goals'] / h2h_games if h2h_games > 0 else 0
        processed_data.append(match_info)
        
        winner = home_team if row['home_score'] > row['away_score'] else (away_team if row['home_score'] < row['away_score'] else 'Draw')
        h2h_history[pair_key].append({'winner': winner, 'home': home_team, 'away': away_team, 'h_score': row['home_score'], 'a_score': row['away_score']})
        
    return pd.DataFrame(processed_data)

def add_misc_features(df, country_year_profile_lookup):
    df = df.copy()
    df['tournament_weight'] = df['tournament'].apply(tournament_weight)
    df['result'] = df.apply(lambda r: 'Home Win' if r['home_score'] > r['away_score'] else ('Away Win' if r['home_score'] < r['away_score'] else 'Draw'), axis=1)
    df['match_year'] = pd.to_datetime(df['date']).dt.year.astype('Int64')
    df['home_team_slug'] = df['home_team'].apply(team_name_to_slug)
    df['away_team_slug'] = df['away_team'].apply(team_name_to_slug)

    df = df.merge(country_year_profile_lookup.rename(columns={'team_slug': 'home_team_slug', 'team_age': 'home_team_age', 'team_height': 'home_team_height'}), 
                  left_on=['home_team_slug', 'match_year'], right_on=['home_team_slug', 'year'], how='left').drop(columns=['year'])
    df = df.merge(country_year_profile_lookup.rename(columns={'team_slug': 'away_team_slug', 'team_age': 'away_team_age', 'team_height': 'away_team_height'}), 
                  left_on=['away_team_slug', 'match_year'], right_on=['away_team_slug', 'year'], how='left').drop(columns=['year'])

    df['glicko_rd_sum'] = df['home_glicko_rd'] + df['away_glicko_rd']
    df['height_difference'] = df['home_team_height'] - df['away_team_height']
    df['home_score_potential'] = df['home_form_GF'] + df['away_form_GA']
    df['away_score_potential'] = df['away_form_GF'] + df['home_form_GA']
    return df.drop(columns=['match_year', 'home_team_slug', 'away_team_slug'])

# --- Glicko-2 Player & Coach Rating System ---

def get_position_multipliers(position):
    pos_map = {
        'Goalkeeper': (0.1, 1.5, 'GK'), 'Centre Back': (0.5, 1.2, 'DF'), 'Defender': (0.5, 1.2, 'DF'), 'Sweeper': (0.5, 1.3, 'DF'),
        'Libero': (0.7, 1.1, 'DF'), 'Left Back': (0.6, 1.1, 'DF'), 'Right Back': (0.6, 1.1, 'DF'), 'Left Wing-Back': (0.8, 1.0, 'DF'),
        'Striker': (1.5, 0.3, 'FW'), 'Centre Forward': (1.5, 0.3, 'FW'), 'Secondary striker': (1.4, 0.4, 'FW'),
        'Attacking Midfielder': (1.3, 0.6, 'MF'), 'Left Midfielder': (1.1, 0.8, 'MF'), 'Right Midfielder': (1.1, 0.8, 'MF'),
        'Centre Midfielder': (1.0, 0.9, 'MF'), 'Midfielder': (1.0, 0.9, 'MF'), 'Defensive Midfielder': (0.8, 1.1, 'MF'),
        'Left Winger': (1.25, 0.5, 'FW'), 'Right Winger': (1.25, 0.5, 'FW'),
    }
    return pos_map.get(str(position).strip(), (1.0, 1.0, 'UNK'))

class PlayerGlicko2:
    def __init__(self, rating=1500, rd=350, vol=0.06, tau=1):
        self.r = (rating - 1500) / 173.7178
        self.rd = rd / 173.7178
        self.sigma = vol
        self.tau = tau
        
    def get_rating(self): return (self.r * 173.7178) + 1500
    def get_rd(self): return self.rd * 173.7178
    def _g(self, phi): return 1 / np.sqrt(1 + 3 * (phi**2) / (np.pi**2))
    def _E(self, mu, mu_j, phi_j): return 1 / (1 + np.exp(-self._g(phi_j) * (mu - mu_j)))

    def update(self, opponent_r, opponent_rd, outcome, multiplier=1.0, t_weight=20):
        mu, phi, sigma, tau = self.r, self.rd, self.sigma, self.tau
        mu_j, phi_j = (opponent_r - 1500) / 173.7178, opponent_rd / 173.7178
        v = 1 / (self._g(phi_j)**2 * self._E(mu, mu_j, phi_j) * (1 - self._E(mu, mu_j, phi_j)))
        delta = v * self._g(phi_j) * (outcome - self._E(mu, mu_j, phi_j))
        
        a = np.log(sigma**2)
        def f(x):
            ex = np.exp(x)
            return (ex * (delta**2 - phi**2 - v - ex)) / (2 * (phi**2 + v + ex)**2) - (x - a) / (tau**2)
            
        A, B = a, (np.log(delta**2 - phi**2 - v) if delta**2 > phi**2 + v else a - tau)
        if delta**2 <= phi**2 + v:
            k = 1
            while f(a - k * tau) < 0: k += 1
            B = a - k * tau
            
        fa, fb = f(A), f(B)
        while abs(B - A) > 1e-6:
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb <= 0: A, fa = B, fb
            else: fa /= 2
            B, fb = C, fc
            
        new_sigma = np.exp(A / 2)
        phi_star = np.sqrt(phi**2 + new_sigma**2)
        new_phi = 1 / np.sqrt(1 / (phi_star**2) + 1 / v)
        rating_step = (new_phi**2) * self._g(phi_j) * (outcome - self._E(mu, mu_j, phi_j))
        self.r = mu + (rating_step * multiplier * (t_weight / 20.0))
        self.rd, self.sigma = new_phi, new_sigma

# --- Final Aggregation Helpers ---

def get_role(position):
    pos_map = {
        'Goalkeeper': 'GK', 'Centre Back': 'DF', 'Defender': 'DF', 'Sweeper': 'DF', 'Libero': 'DF',
        'Left Back': 'DF', 'Right Back': 'DF', 'Left Wing-Back': 'DF', 'Right Wing-Back': 'DF',
        'Striker': 'FW', 'Centre Forward': 'FW', 'Secondary striker': 'FW', 'Left Winger': 'FW', 'Right Winger': 'FW',
        'Attacking Midfielder': 'MF', 'Left Midfielder': 'MF', 'Right Midfielder': 'MF',
        'Centre Midfielder': 'MF', 'Midfielder': 'MF', 'Defensive Midfielder': 'MF'
    }
    return pos_map.get(str(position).strip(), 'UNK')

def merge_and_ffill(base_df, feature_df, prefix):
    merged = pd.merge(base_df, feature_df, left_on=['date', prefix+'_team'], right_on=['date', 'team'], how='left').drop(columns=['team'])
    feat_cols = [c for c in feature_df.columns if c not in ['date', 'team']]
    merged.rename(columns={c: f'{prefix}_{c}' for c in feat_cols}, inplace=True)
    target_cols = [f'{prefix}_{c}' for c in feat_cols]
    merged[target_cols] = merged.groupby(prefix+'_team', group_keys=False)[target_cols].apply(lambda x: x.ffill())
    merged[target_cols] = merged[target_cols].fillna(1500)
    return merged

# --- Main Engine ---

def main():
    script_dir = Path(__file__).parent
    # 1. Setup & Data Loading
    api = setup_kaggle()
    # Path to the shared 'data' folder in the project root
    dataset_path = (script_dir.parent / 'data').resolve()
    download_data(api, dataset_path)
    
    df = pd.read_csv(dataset_path / 'results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Load profile data
    profile_path = (script_dir / "downloaded_files/scrape_exports/afc_country_year_data.csv").resolve()
    
    if not profile_path.exists():
        print(f"Warning: {profile_path} not found. Misc features may be limited.")
        country_year_profile_lookup = pd.DataFrame(columns=["team_slug", "year", "team_age", "team_height"])
    else:
        country_year_stats_df = pd.read_csv(profile_path)
        country_year_stats_df["year"] = pd.to_numeric(country_year_stats_df["year"], errors="coerce").astype("Int64")
        country_year_stats_df["team_slug"] = country_year_stats_df["country_slug"].astype(str)
        country_year_stats_df["team_age"] = pd.to_numeric(country_year_stats_df["average_age"], errors="coerce")
        country_year_stats_df["team_height"] = country_year_stats_df["average_height"].apply(parse_height_to_m)
        country_year_profile_lookup = country_year_stats_df[["team_slug", "year", "team_age", "team_height"]].drop_duplicates(["team_slug", "year"])

    # 2. Filtering
    teams_in_asia = ["Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China PR", "Taiwan", "North Korea", "Guam", "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "Northern Mariana Islands", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen"]
    afc_matches = df[(df['home_team'].isin(teams_in_asia)) & (df['away_team'].isin(teams_in_asia)) & (df['date'].dt.year >= 1995)].copy()
    
    # Forfeited logic
    mask = (afc_matches['date'].dt.year >= 2025) & (afc_matches['date'].dt.month <= 9) & ((afc_matches['home_team'] == 'Malaysia') | (afc_matches['away_team'] == 'Malaysia'))
    afc_matches.loc[mask, 'tournament'] = 'Forfeited'

    # 3. Glicko & Form Features
    afc_with_glicko, final_glicko_obj = get_pre_match_glicko(afc_matches, teams_in_asia)
    afc_with_form, final_form_history = add_form_features(afc_with_glicko, teams_in_asia)
    afc_with_h2h = add_h2h_features(afc_with_form)
    afc_final = add_misc_features(afc_with_h2h, country_year_profile_lookup)

    # 4. Player & Coach Ratings
    lineups_path = (script_dir / "downloaded_files/scrape_exports/afc_match_lineups_coaches.csv").resolve()
    players_path = (script_dir / "downloaded_files/scrape_exports/afc_players_data.csv").resolve()

    if not lineups_path.exists() or not players_path.exists():
        raise FileNotFoundError(f"Match lineups ({lineups_path}) or players data ({players_path}) not found.")

    lineups_df = pd.read_csv(lineups_path)
    players_df = pd.read_csv(players_path)
    merged_lineups = pd.merge(lineups_df, players_df, on='player_link', how='left')
    merged_lineups['date'] = pd.to_datetime(merged_lineups['date'])
    
    # Perspectives for training
    home_p = afc_with_form[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'home_glicko_rating', 'away_glicko_rating', 'neutral', 'tournament']].copy()
    home_p.columns = ['date', 'team', 'opponent', 'team_goals', 'opponent_goals', 'team_glicko_rating', 'opponent_glicko_rating', 'neutral', 'tournament']
    home_p['ground'] = home_p['neutral'].apply(lambda x: 'neutral' if x else 'home')
    away_p = afc_with_form[['date', 'away_team', 'home_team', 'away_score', 'home_score', 'away_glicko_rating', 'home_glicko_rating', 'neutral', 'tournament']].copy()
    away_p.columns = ['date', 'team', 'opponent', 'team_goals', 'opponent_goals', 'team_glicko_rating', 'opponent_glicko_rating', 'neutral', 'tournament']
    away_p['ground'] = away_p['neutral'].apply(lambda x: 'neutral' if x else 'away')
    res_p = pd.concat([home_p, away_p], ignore_index=True)
    res_p['result'] = res_p.apply(lambda r: 'Win' if r['team_goals'] > r['opponent_goals'] else ('Loss' if r['team_goals'] < r['opponent_goals'] else 'Draw'), axis=1)

    # Player updates
    chrono_players = pd.merge(merged_lineups, res_p, on=['date', 'team'], how='inner').sort_values('date')
    player_ratings = {}
    player_history = []
    outcome_map = {'Win': 1.0, 'Draw': 0.5, 'Loss': 0.0}

    for _, row in chrono_players.iterrows():
        p_link = row['player_link']
        if p_link not in player_ratings: player_ratings[p_link] = PlayerGlicko2(rating=1500)
        p_obj = player_ratings[p_link]
        off_m, def_m, p_group = get_position_multipliers(row['position'])
        outcome = outcome_map.get(row['result'], 0.5)
        
        pos_a = (row.get('goals', 0) * 0.2 * off_m) + (row.get('assists', 0) * 0.1 * off_m) + (0.2 * def_m if row['opponent_goals'] == 0 else 0)
        neg_a = (row.get('yellow_card', 0) * 0.1) + (row.get('red_card', 0) * 0.3) + (max(0, row['opponent_goals'] - 1) * 0.05 * def_m)
        adj = 1.0 + (pos_a - neg_a)
        
        pre_r = p_obj.get_rating()
        p_obj.update(row['opponent_glicko_rating'], 100, outcome, multiplier=max(0.5, min(2.0, adj)), t_weight=tournament_weight(row['tournament']))
        player_history.append({'date': row['date'], 'team': row['team'], 'player_link': p_link, 'position': row['position'], 'pre_rating': pre_r})

    history_df = pd.DataFrame(player_history)

    # Coach updates
    coach_events = lineups_df[['date', 'team', 'coach']].dropna(subset=['coach']).drop_duplicates()
    coach_events['date'] = pd.to_datetime(coach_events['date'])
    coach_res = pd.merge(coach_events, res_p, on=['date', 'team'], how='inner').sort_values('date')
    coach_ratings = {}
    coach_history = []

    for _, row in coach_res.iterrows():
        c_name = row['coach']
        if c_name not in coach_ratings: coach_ratings[c_name] = PlayerGlicko2(rating=1500)
        c_obj = coach_ratings[c_name]
        outcome = outcome_map.get(row['result'], 0.5)
        adj = 1.0 + (0.05 * row['team_goals'] + (0.2 if row['opponent_goals'] == 0 else 0)) - (0.05 * max(0, row['opponent_goals'] - 1))
        pre_r = c_obj.get_rating()
        c_obj.update(row['opponent_glicko_rating'], 100, outcome, multiplier=max(0.5, min(2.0, adj)), t_weight=tournament_weight(row['tournament']))
        coach_history.append({'date': row['date'], 'team': row['team'], 'pre_rating': pre_r})

    coach_history_df = pd.DataFrame(coach_history)

    # 5. Aggregate and Poisson DF
    history_df['role'] = history_df['position'].apply(get_role)
    p_agg = history_df.groupby(['date', 'team'])['pre_rating'].agg(['mean', 'std', 'count']).reset_index()
    p_agg.columns = ['date', 'team', 'avg_player_rating', 'std_player_rating', 'player_count']
    r_agg = history_df.groupby(['date', 'team', 'role'])['pre_rating'].mean().unstack(fill_value=np.nan).reset_index()
    r_agg.columns = ['date', 'team'] + [f'avg_{r}_rating' for r in r_agg.columns if r not in ['date', 'team']]
    c_agg = coach_history_df.rename(columns={'pre_rating': 'coach_rating'})

    afc_poisson_df = afc_final.copy()
    afc_poisson_df['date'] = pd.to_datetime(afc_poisson_df['date'])
    for side in ['home', 'away']:
        afc_poisson_df = merge_and_ffill(afc_poisson_df, p_agg, side)
        afc_poisson_df = merge_and_ffill(afc_poisson_df, r_agg, side)
        afc_poisson_df = merge_and_ffill(afc_poisson_df, c_agg, side)

    # Clean missing data
    rating_suffixes = ['avg_player_rating', 'std_player_rating', 'player_count', 'avg_DF_rating', 'avg_FW_rating', 'avg_GK_rating', 'avg_MF_rating', 'avg_UNK_rating', 'coach_rating']
    for side in ['home', 'away']:
        mask = afc_poisson_df[f'{side}_avg_player_rating'] == 1500
        for s in rating_suffixes:
            if f'{side}_{s}' in afc_poisson_df.columns: afc_poisson_df.loc[mask, f'{side}_{s}'] = np.nan

    afc_poisson_df['player_rating_difference'] = afc_poisson_df['home_avg_player_rating'] - afc_poisson_df['away_avg_player_rating']
    afc_poisson_df['coach_rating_difference'] = afc_poisson_df['home_coach_rating'] - afc_poisson_df['away_coach_rating']

    # 6. Training
    train_mask = (afc_poisson_df['date'].dt.year >= 1996) & (afc_poisson_df['date'].dt.year <= 2017)
    val_mask = (afc_poisson_df['date'].dt.year >= 2018) & (afc_poisson_df['date'].dt.year <= 2020)
    
    exclude_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'city', 'country', 'result', 'home_glicko_rating_post', 'away_glicko_rating_post', 'away_form_games_max10', 'home_form_games_max10', 'H2H_games_max5', 'home_glicko_rd', 'away_glicko_rd', 'home_form_GF', 'home_form_GA', 'away_form_GF', 'away_form_GA', 'home_team_age', 'away_team_age', 'home_form_streak', 'away_form_streak', 'predicted_result', 'home_player_count', 'away_player_count']
    features = [c for c in afc_poisson_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    X_train, y_train_h, y_train_a = afc_poisson_df.loc[train_mask, features], afc_poisson_df.loc[train_mask, 'home_score'], afc_poisson_df.loc[train_mask, 'away_score']
    X_val, y_val_h, y_val_a = afc_poisson_df.loc[val_mask, features], afc_poisson_df.loc[val_mask, 'home_score'], afc_poisson_df.loc[val_mask, 'away_score']

    params = {'objective': 'count:poisson', 'eval_metric': 'poisson-nloglik', 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'early_stopping_rounds': 50, 'random_state': 42}
    model_h = xgb.XGBRegressor(**params, n_estimators=1000)
    model_a = xgb.XGBRegressor(**params, n_estimators=1000)

    print("Training models...")
    model_h.fit(X_train, y_train_h, eval_set=[(X_val, y_val_h)], verbose=False)
    model_a.fit(X_train, y_train_a, eval_set=[(X_val, y_val_a)], verbose=False)

    # Optional Evaluation on Test Set
    test_mask = (afc_poisson_df['date'].dt.year >= 2021) & (afc_poisson_df['date'].dt.year <= 2026)
    if test_mask.any():
        X_test = afc_poisson_df.loc[test_mask, features]
        y_test_h = afc_poisson_df.loc[test_mask, 'home_score']
        y_test_a = afc_poisson_df.loc[test_mask, 'away_score']
        
        preds_h = model_h.predict(X_test)
        preds_a = model_a.predict(X_test)
        
        # Simple accuracy check (could be expanded with Dixon-Coles)
        pred_results = ['Home Win' if h > a else ('Away Win' if a > h else 'Draw') for h, a in zip(preds_h, preds_a)]
        true_results = afc_poisson_df.loc[test_mask, 'result']
        accuracy = (pd.Series(pred_results, index=true_results.index) == true_results).mean()
        print(f"Poisson Model Accuracy on Test Set (2021-2026): {accuracy:.2%}")

    # 7. Export
    model_data = {
        'model_home': model_h,
        'model_away': model_a,
        'features': features,
        'teams': teams_in_asia,
        'player_ratings': player_ratings,
        'coach_ratings': coach_ratings,
        'glicko_obj': final_glicko_obj,
        'team_history': final_form_history,
        'country_year_profile_lookup': country_year_profile_lookup
    }
    joblib.dump(model_data, 'afc_football_model.joblib')
    print("Model saved to afc_football_model.joblib")

if __name__ == "__main__":
    main()
