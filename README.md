# FootballPredictionsAFC
A Python Notebook for predictng football results in the AFC region.

## Overview

FootballPredictionsAFC is a comprehensive toolkit for predicting Asian Football Confederation (AFC) match outcomes using advanced statistical modeling techniques combined with web scraping capabilities.

### Website links

- GitHub: https://github.com/quachthanhhung12349/FootballPredictionsAFC
- Website (Includes per-game prediction and tournament simulation): https://football-predictions-afc--quachthanhhung1.replit.app
---

## Components

### 1. Predict Asian Football Matches (2000-2026).ipynb

A sophisticated prediction model that forecasts AFC football match outcomes using an advanced statistical pipeline.

**Key Features:**
- **Glicko-1 Rating System**: Implements the Glicko-1 rating model to track and update team strengths dynamically based on match results and rating uncertainties
- **Poisson Regression with Dixon-Coles Adjustments**: Uses Poisson regression to model goal scoring with Dixon-Coles adjustments to better capture low-scoring outcomes and dependencies between goals
- **Advanced Data Engineering**:
	- Rating differences between competing teams
	- Head-to-head (H2H) historical records and matchup dynamics
	- Team form records spanning multiple match periods
	- Player and coach data, including the rating approximation of the players/coaches and the overall rating of the squad overall
- **Comprehensive Pipeline**: Integrates data preprocessing, feature engineering, model training, and prediction generation into a unified workflow

**Output**: Probabilistic predictions for AFC match outcomes over the 2000-2026 period

### 2. AFC Games Scraper

A web scraper for collecting comprehensive football match data from AFC regions.

**Data Source:**
- Scrapes data from [national-football-teams.com](https://national-football-teams.com/), one of the few remaining scraping-friendly football data websites
- Other popular sources (FBRef, Transfermarkt) have implemented strong security measures that make scraping prohibitively difficult

**Current Capabilities:**
- Collect historical match results and team data
- Prepare datasets for the prediction model

**Future Enhancements:**
- Scraping coach data for coaching impact analysis
- Scraping player-level data for more granular performance metrics


