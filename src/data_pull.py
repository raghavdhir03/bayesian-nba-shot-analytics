from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats, commonteamroster
from nba_api.stats.static import players, teams
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path

# CONFIGURATION

SEASON = '2023-24'
SEASON_TYPE = 'Regular Season'
OUTPUT_DIR = Path('data')
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_DATA_FILE = OUTPUT_DIR / 'nba_shots_2023_24.parquet'
POSITION_DATA_FILE = OUTPUT_DIR / 'player_positions_2023_24.parquet'
LEAGUE_PRIORS_FILE = OUTPUT_DIR / 'league_priors_2023_24.parquet'
POSITION_PRIORS_FILE = OUTPUT_DIR / 'position_priors_2023_24.parquet'
PLAYER_STATS_FILE = OUTPUT_DIR / 'player_shot_stats_2023_24.parquet'

RATE_LIMIT_DELAY = 0.6  # seconds between requests
RETRY_DELAY = 5  # seconds to wait on error


# STEP 1: SCRAPE POSITION DATA

def scrape_position_data(season=SEASON):
    
    print(f"\n{'='*70}")
    print(f"SCRAPING PLAYER POSITIONS - {season}")
    print(f"{'='*70}\n")
    
    nba_teams = teams.get_teams()
    team_ids = [team['id'] for team in nba_teams]
    
    print(f"📋 Found {len(team_ids)} NBA teams")
    
    rows = []
    failed_teams = []
    
    for team_id in tqdm(team_ids, desc="Scraping rosters"):
        try:
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id, 
                season=season
            )
            player_data = roster.get_normalized_dict()['CommonTeamRoster']
            
            for player in player_data:
                rows.append({
                    'PLAYER_ID': player['PLAYER_ID'], 
                    'PLAYER_NAME': player['PLAYER'],
                    'POSITION': player['POSITION']
                })
                
        except Exception as e:
            tqdm.write(f"Error for team ID {team_id}: {e}")
            failed_teams.append(team_id)
            time.sleep(RETRY_DELAY)
            continue
        
        time.sleep(RATE_LIMIT_DELAY)
    
    player_df = pd.DataFrame(rows)
    
    # Remove duplicates (players traded mid-season will appear on multiple rosters)
    # Keep first occurrence
    player_df = player_df.drop_duplicates(subset='PLAYER_ID', keep='first')
    
    print(f"\n{'='*70}")
    print(f"POSITION SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"Total players with positions: {len(player_df):,}")
    print(f"Failed teams: {len(failed_teams)}")
    
    if failed_teams:
        print(f"\nFailed team IDs: {failed_teams}")
    
    return player_df


def map_position(pos):
    if pd.isna(pos):
        return 'Unknown'
    
    pos = str(pos).upper().strip()
    
    # Guards (includes guard-forward hybrids - more perimeter oriented)
    if 'G' in pos:
        return 'Guard'
    # Centers (pure centers only)
    elif pos == 'C':
        return 'Center'
    # Forwards (includes forward-center hybrids)
    elif 'F' in pos or 'C' in pos:
        return 'Forward'
    else:
        return 'Unknown'



# STEP 2: SCRAPE SHOT DATA
def scrape_shot_data():
    
    print(f"\n{'='*70}")
    print(f"SCRAPING NBA SHOT DATA - {SEASON} {SEASON_TYPE}")
    print(f"{'='*70}\n")
    
    # Get all players who played in the season
    print("Fetching player list...")
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE
    )
    
    players_df = player_stats.get_data_frames()[0]
    players_df = players_df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
    player_ids = list(players_df['PLAYER_ID'])
    
    print(f"✅ Found {len(player_ids)} players\n")
    
    # Scrape shot data
    all_shots = pd.DataFrame()
    failed_players = []
    
    for player_id in tqdm(player_ids, desc="Scraping shots"):
        try:
            response = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_type_all_star=SEASON_TYPE,
                season_nullable=SEASON,
                context_measure_simple='FGA'
            )
            player_df = response.get_data_frames()[0]
            
            # Skip if no shot data
            if player_df.empty:
                continue
            
            all_shots = pd.concat([all_shots, player_df], ignore_index=True)
            
        except Exception as e:
            tqdm.write(f" Error for player ID {player_id}: {e}")
            failed_players.append(player_id)
            time.sleep(RETRY_DELAY)
            continue
        
        time.sleep(RATE_LIMIT_DELAY)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"Total shots scraped: {len(all_shots):,}")
    print(f"Unique players: {all_shots['PLAYER_ID'].nunique()}")
    print(f"Failed players: {len(failed_players)}")
    
    if failed_players:
        print(f"\nFailed player IDs: {failed_players[:10]}..." if len(failed_players) > 10 else failed_players)
    
    return all_shots


# STEP 3: MERGE POSITION DATA WITH SHOTS
def merge_position_data(shots_df, positions_df):
    """Merge position information with shot data"""
    
    print(f"\n{'='*70}")
    print(f"MERGING POSITION DATA")
    print(f"{'='*70}\n")
    
    # Apply position mapping
    positions_df['POSITION'] = positions_df['POSITION'].apply(map_position)
    
    # Show position distribution before merge
    print("Position distribution:")
    print(positions_df['POSITION'].value_counts())
    print()
    
    # Merge on PLAYER_ID
    merged = shots_df.merge(
        positions_df[['PLAYER_ID', 'POSITION']], 
        on='PLAYER_ID', 
        how='left'
    )
    
    # Check for missing positions
    missing_positions = merged['POSITION'].isna().sum()
    print(f"Shots without position data: {missing_positions:,} ({missing_positions/len(merged):.1%})")
    
    # Fill missing positions with 'Unknown'
    merged['POSITION'] = merged['POSITION'].fillna('Unknown')
    
    print(f"✅ Merged dataset shape: {merged.shape}")
    print(f"✅ Total shots with positions: {len(merged):,}")
    
    return merged



# STEP 4: DATA QUALITY CHECKS
def check_data_quality(df):
    """Perform data quality checks and print summary statistics"""
    
    print(f"\n{'='*70}")
    print(f"DATA QUALITY CHECK")
    print(f"{'='*70}\n")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    # Missing values
    print("Missing values by column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values!")
    
    # Key columns check
    required_cols = ['LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG', 'SHOT_ZONE_BASIC', 
                     'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'PLAYER_ID', 'PLAYER_NAME',
                     'POSITION']
    
    print(f"\nRequired columns present: {all(col in df.columns for col in required_cols)}")
    
    # Zone distribution
    print("\nShot distribution by zone:")
    print(df['SHOT_ZONE_BASIC'].value_counts())
    
    # Position distribution in shot data
    print("\nShot distribution by position:")
    print(df['POSITION'].value_counts())
    
    # Overall shooting percentage
    overall_fg = df['SHOT_MADE_FLAG'].mean()
    print(f"\n📊 Overall FG%: {overall_fg:.1%}")
    
    # Shooting percentage by position
    print("\nShooting percentage by position:")
    pos_shooting = df.groupby('POSITION')['SHOT_MADE_FLAG'].agg(['mean', 'count'])
    pos_shooting.columns = ['FG%', 'Attempts']
    print(pos_shooting)
    
    return True



# STEP 5: COMPUTE LEAGUE-WIDE PRIORS (Original - for comparison)
def compute_league_priors(df):
    """Compute league-average shooting stats by zone for Bayesian priors"""
    
    print(f"\n{'='*70}")
    print(f"COMPUTING LEAGUE-WIDE PRIORS")
    print(f"{'='*70}\n")
    
    # Group by zone and compute shooting stats
    priors = df.groupby('SHOT_ZONE_BASIC').agg({
        'SHOT_MADE_FLAG': ['sum', 'count', 'mean']
    }).reset_index()
    
    # Flatten column names
    priors.columns = ['zone', 'makes', 'attempts', 'fg_pct']
    
    # Add Beta prior parameters (for Beta-Binomial conjugate prior)
    # α (alpha) = makes, β (beta) = misses
    priors['alpha'] = priors['makes']
    priors['beta'] = priors['attempts'] - priors['makes']
    
    # Sort by volume
    priors = priors.sort_values('attempts', ascending=False)
    
    print("League-wide shooting by zone:")
    print(priors.to_string(index=False))
    print(f"\nPriors computed for {len(priors)} zones")
    
    return priors


# STEP 6: COMPUTE POSITION-SPECIFIC PRIORS (New - Main approach)
def compute_position_priors(df):
 
    
    print(f"\n{'='*70}")
    print(f"COMPUTING POSITION-SPECIFIC PRIORS")
    print(f"{'='*70}\n")
    
    # Exclude Unknown positions from prior calculation
    df_known = df[df['POSITION'] != 'Unknown'].copy()
    
    # Group by position AND zone
    priors = df_known.groupby(['POSITION', 'SHOT_ZONE_BASIC']).agg({
        'SHOT_MADE_FLAG': ['sum', 'count', 'mean']
    }).reset_index()
    
    # Flatten column names
    priors.columns = ['position', 'zone', 'makes', 'attempts', 'fg_pct']
    
    # Add Beta prior parameters
    priors['alpha'] = priors['makes']
    priors['beta'] = priors['attempts'] - priors['makes']
    
    # Sort by position then attempts
    priors = priors.sort_values(['position', 'attempts'], ascending=[True, False])
    
    # Display by position
    for position in ['Guard', 'Forward', 'Center']:
        pos_priors = priors[priors['position'] == position]
        print(f"\n{position.upper()} PRIORS:")
        print(pos_priors[['zone', 'makes', 'attempts', 'fg_pct']].to_string(index=False))
    
    print(f"\nPosition-specific priors computed: {len(priors)} combinations")
    print(f"   (3 positions × {df['SHOT_ZONE_BASIC'].nunique()} zones)")
    
    # Summary comparison
    print("\n" + "="*70)
    print("POSITION COMPARISON - Above the Break 3PT%")
    print("="*70)
    atb3 = priors[priors['zone'] == 'Above the Break 3']
    print(atb3[['position', 'fg_pct', 'attempts']].to_string(index=False))
    
    return priors



# STEP 7: COMPUTE PER-PLAYER STATISTICS
def compute_player_stats(df):
    """Compute per-player shooting statistics by zone with position info"""
    
    print(f"\n{'='*70}")
    print(f"COMPUTING PLAYER STATISTICS")
    print(f"{'='*70}\n")
    
    # Group by player, position, and zone
    player_stats = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'SHOT_ZONE_BASIC']).agg({
        'SHOT_MADE_FLAG': ['sum', 'count', 'mean']
    }).reset_index()
    
    # Flatten columns
    player_stats.columns = ['player_id', 'player_name', 'position', 'zone', 
                            'makes', 'attempts', 'raw_fg_pct']
    
    # Filter out very low-volume zones
    MIN_ATTEMPTS = 5
    player_stats = player_stats[player_stats['attempts'] >= MIN_ATTEMPTS]
    
    print(f"Player-zone combinations: {len(player_stats):,}")
    print(f" Unique players with stats: {player_stats['player_id'].nunique()}")
    print(f"   (Filtered to zones with ≥{MIN_ATTEMPTS} attempts)")
    
    # Show examples by position
    print("\n" + "="*70)
    print("EXAMPLE: Top 3PT Shooters by Position (Above the Break 3)")
    print("="*70)
    
    atb3 = player_stats[player_stats['zone'] == 'Above the Break 3'].copy()
    
    for position in ['Guard', 'Forward', 'Center']:
        print(f"\n{position}s:")
        pos_shooters = atb3[atb3['position'] == position].copy()
        pos_shooters = pos_shooters.sort_values('raw_fg_pct', ascending=False).head(5)
        print(pos_shooters[['player_name', 'makes', 'attempts', 'raw_fg_pct']].to_string(index=False))
    
    return player_stats



# STEP 8: SAVE ALL DATA

def save_data(shots_df, positions_df, league_priors_df, position_priors_df, player_stats_df):
    """Save processed data to disk"""
    
    print(f"\n{'='*70}")
    print(f"SAVING DATA")
    print(f"{'='*70}\n")
    
    # Save raw shots (now includes position)
    shots_df.to_parquet(RAW_DATA_FILE, index=False)
    print(f"Raw shots saved to: {RAW_DATA_FILE}")
    
    # Save position data
    positions_df.to_parquet(POSITION_DATA_FILE, index=False)
    print(f"Position data saved to: {POSITION_DATA_FILE}")
    
    # Save league-wide priors
    league_priors_df.to_parquet(LEAGUE_PRIORS_FILE, index=False)
    print(f"League priors saved to: {LEAGUE_PRIORS_FILE}")
    
    # Save position-specific priors
    position_priors_df.to_parquet(POSITION_PRIORS_FILE, index=False)
    print(f"Position priors saved to: {POSITION_PRIORS_FILE}")
    
    # Save player stats
    player_stats_df.to_parquet(PLAYER_STATS_FILE, index=False)
    print(f"Player stats saved to: {PLAYER_STATS_FILE}")
    
    print(f"\ntotal disk usage: {sum(f.stat().st_size for f in OUTPUT_DIR.glob('*.parquet')) / 1024**2:.2f} MB")



# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution pipeline"""
    
    print(f"\n{'#'*70}")
    print(f"# NBA BAYESIAN SHOT ANALYTICS - DATA PIPELINE v2")
    print(f"# Position-Specific Priors")
    print(f"# Season: {SEASON} {SEASON_TYPE}")
    print(f"{'#'*70}")
    
    if POSITION_DATA_FILE.exists():
        print(f"\nFound existing position data at {POSITION_DATA_FILE}")
        response = input("Do you want to re-scrape positions? (y/N): ").strip().lower()
        
        if response == 'y':
            positions_df = scrape_position_data()
        else:
            print("📂 Loading existing position data...")
            positions_df = pd.read_parquet(POSITION_DATA_FILE)
            print(f"Loaded {len(positions_df):,} player positions from disk")
    else:
        positions_df = scrape_position_data()
    

    if RAW_DATA_FILE.exists():
        print(f"\nFound existing shot data at {RAW_DATA_FILE}")
        response = input("Do you want to re-scrape shots? (y/N): ").strip().lower()
        
        if response == 'y':
            shots_df = scrape_shot_data()
            # Merge position data
            shots_df = merge_position_data(shots_df, positions_df)
        else:
            print("Loading existing shot data...")
            shots_df = pd.read_parquet(RAW_DATA_FILE)
            print(f"Loaded {len(shots_df):,} shots from disk")
            
            # Check if position column exists
            if 'POSITION' not in shots_df.columns:
                print("\nPosition data not in shots file, merging now...")
                shots_df = merge_position_data(shots_df, positions_df)
    else:
        shots_df = scrape_shot_data()
        shots_df = merge_position_data(shots_df, positions_df)
    
    # ========================================================================
    # STEP 3: Data quality checks
    # ========================================================================
    check_data_quality(shots_df)
    
    # ========================================================================
    # STEP 4: Compute priors
    # ========================================================================
    league_priors = compute_league_priors(shots_df)
    position_priors = compute_position_priors(shots_df)
    
    # ========================================================================
    # STEP 5: Compute player statistics
    # ========================================================================
    player_stats = compute_player_stats(shots_df)
    
    # ========================================================================
    # STEP 6: Save everything
    # ========================================================================
    save_data(shots_df, positions_df, league_priors, position_priors, player_stats)
    
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE!")
    print(f"{'='*70}\n")
    print("Next steps:")
    print("1. Run Bayesian modeling script with position-specific priors")
    print("2. Compare league-wide vs position-specific posteriors")
    print("3. Build visualization dashboard with position filters")
    print("4. Analyze shrinkage effects by position\n")


if __name__ == "__main__":
    main()