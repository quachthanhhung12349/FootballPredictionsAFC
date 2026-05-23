import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path('FootballPredictionsAFC').resolve()))
import train_afc_model

def get_latest_stats():
    dataset_path = Path('data')
    df = pd.read_csv(dataset_path / 'results.csv')
    df['date'] = pd.to_datetime(df['date'])
    # Load profile data
    profile_path = Path("FootballPredictionsAFC/downloaded_files/scrape_exports/afc_country_year_data.csv")
    if profile_path.exists():
        country_year_stats_df = pd.read_csv(profile_path)
        country_year_stats_df["year"] = pd.to_numeric(country_year_stats_df["year"], errors="coerce").astype("Int64")
        country_year_stats_df["team_slug"] = country_year_stats_df["country_slug"].astype(str)
        country_year_stats_df["team_age"] = pd.to_numeric(country_year_stats_df["average_age"], errors="coerce")
        country_year_stats_df["team_height"] = country_year_stats_df["average_height"].apply(train_afc_model.parse_height_to_m)
        country_year_profile_lookup = country_year_stats_df[["team_slug", "year", "team_age", "team_height"]].drop_duplicates(["team_slug", "year"])
    else:
        country_year_profile_lookup = pd.DataFrame(columns=["team_slug", "year", "team_age", "team_height"])

    teams_in_asia = ["Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China PR", "Taiwan", "North Korea", "Guam", "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "Northern Mariana Islands", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen"]
    afc_matches = df[(df['home_team'].isin(teams_in_asia)) & (df['away_team'].isin(teams_in_asia)) & (df['date'].dt.year >= 1995)].copy()

    afc_with_glicko = train_afc_model.get_pre_match_glicko(afc_matches, teams_in_asia)
    afc_with_form = train_afc_model.add_form_features(afc_with_glicko, teams_in_asia)
    afc_with_h2h = train_afc_model.add_h2h_features(afc_with_form)
    afc_final = train_afc_model.add_misc_features(afc_with_h2h, country_year_profile_lookup)

    lineups_path = Path("FootballPredictionsAFC/downloaded_files/scrape_exports/afc_match_lineups_coaches.csv")
    lineups_df = pd.read_csv(lineups_path)
    
    # We just need the final states!
    # Glicko
    latest_glicko = {}
    for _, row in afc_final.sort_values('date').iterrows():
        latest_glicko[row['home_team']] = {'rating': row['home_glicko_rating_post'], 'rd': row['home_glicko_rd']} # approximations, rd should be updated but rd from previous is close enough
        latest_glicko[row['away_team']] = {'rating': row['away_glicko_rating_post'], 'rd': row['away_glicko_rd']}

    return afc_final, lineups_df

if __name__ == '__main__':
    get_latest_stats()
