"""
NBA Bayesian Posterior Computation
Computes position-specific Bayesian posteriors with credible intervals
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = Path('data')
POSITION_PRIORS_FILE = OUTPUT_DIR / 'position_priors_2023_24.parquet'
PLAYER_STATS_FILE = OUTPUT_DIR / 'player_shot_stats_2023_24.parquet'
OUTPUT_FILE = OUTPUT_DIR / 'bayesian_posteriors_2023_24.parquet'


CREDIBLE_INTERVAL = 0.95


# ============================================================================
# BAYESIAN POSTERIOR COMPUTATION
# ============================================================================
def compute_posterior(player_makes, player_attempts, prior_alpha, prior_beta):
    # Posterior parameters
    posterior_alpha = prior_alpha + player_makes
    posterior_beta = prior_beta + (player_attempts - player_makes)
    
    # Posterior mean (expected value)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    # Credible interval
    alpha_level = (1 - CREDIBLE_INTERVAL) / 2
    ci_lower = stats.beta.ppf(alpha_level, posterior_alpha, posterior_beta)
    ci_upper = stats.beta.ppf(1 - alpha_level, posterior_alpha, posterior_beta)
    
    return posterior_alpha, posterior_beta, posterior_mean, ci_lower, ci_upper


def compute_shrinkage(raw_pct, posterior_mean):
    return raw_pct - posterior_mean


# ============================================================================
# MAIN COMPUTATION
# ============================================================================
def compute_all_posteriors():
    
    print(f"\n{'='*70}")
    print(f"COMPUTING BAYESIAN POSTERIORS")
    print(f"Using Position-Specific Priors")
    print(f"{'='*70}\n")
    
    # Load data
    print("📂 Loading data...")
    priors = pd.read_parquet(POSITION_PRIORS_FILE)
    player_stats = pd.read_parquet(PLAYER_STATS_FILE)
    
    print(f"✅ Loaded {len(priors)} position-zone priors")
    print(f"✅ Loaded {len(player_stats):,} player-zone statistics")
    
    # Merge priors with player stats on position and zone
    print("\n🔗 Merging priors with player data...")
    merged = player_stats.merge(
        priors[['position', 'zone', 'alpha', 'beta', 'fg_pct']],
        left_on=['position', 'zone'],
        right_on=['position', 'zone'],
        how='left',
        suffixes=('', '_prior')
    )
    
    # Check for missing priors
    missing = merged['alpha'].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} player-zone combinations without matching priors")
        print("   (These are likely 'Unknown' position players)")
        # For now, drop these - in production you'd use league-wide prior as fallback
        merged = merged.dropna(subset=['alpha', 'beta'])
    
    print(f"✅ Matched {len(merged):,} player-zone combinations with priors")
    
    # Compute posteriors
    print("\n🧮 Computing posteriors...")
    posteriors = []
    
    for idx, row in merged.iterrows():
        post_alpha, post_beta, post_mean, ci_low, ci_high = compute_posterior(
            player_makes=row['makes'],
            player_attempts=row['attempts'],
            prior_alpha=row['alpha'],
            prior_beta=row['beta']
        )
        
        posteriors.append({
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'position': row['position'],
            'zone': row['zone'],
            'attempts': row['attempts'],
            'makes': row['makes'],
            'raw_fg_pct': row['raw_fg_pct'],
            'prior_fg_pct': row['fg_pct'],
            'posterior_mean': post_mean,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'ci_width': ci_high - ci_low,
            'shrinkage': compute_shrinkage(row['raw_fg_pct'], post_mean),
            'prior_alpha': row['alpha'],
            'prior_beta': row['beta'],
            'posterior_alpha': post_alpha,
            'posterior_beta': post_beta
        })
    
    results_df = pd.DataFrame(posteriors)
    
    print(f"✅ Computed {len(results_df):,} Bayesian posteriors")
    
    return results_df


# ============================================================================
# ANALYSIS & INSIGHTS
# ============================================================================
def analyze_results(df):
    
    print(f"\n{'='*70}")
    print(f"BAYESIAN ANALYSIS INSIGHTS")
    print(f"{'='*70}\n")
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"  Average shrinkage: {df['shrinkage'].abs().mean():.3f}")
    print(f"  Average CI width: {df['ci_width'].mean():.3f}")
    print(f"  Median attempts: {df['attempts'].median():.0f}")
    
    # Shrinkage vs Sample Size

    print(f"\n{'='*70}")
    print("SHRINKAGE BY SAMPLE SIZE")
    print(f"{'='*70}")
    
    # Categorize by sample size
    df['sample_category'] = pd.cut(
        df['attempts'], 
        bins=[0, 20, 50, 100, 500, np.inf],
        labels=['Very Low (≤20)', 'Low (21-50)', 'Medium (51-100)', 
                'High (101-500)', 'Very High (>500)']
    )
    
    shrinkage_by_sample = df.groupby('sample_category').agg({
        'shrinkage': ['mean', 'std'],
        'ci_width': 'mean',
        'attempts': ['count', 'mean']
    }).round(4)
    
    print(shrinkage_by_sample)
    
    # Position-Specific Results
    print(f"\n{'='*70}")
    print("RESULTS BY POSITION")
    print(f"{'='*70}")
    
    position_summary = df.groupby('position').agg({
        'posterior_mean': 'mean',
        'shrinkage': lambda x: x.abs().mean(),
        'ci_width': 'mean',
        'player_id': 'nunique'
    }).round(4)
    
    position_summary.columns = ['Avg Posterior FG%', 'Avg |Shrinkage|', 
                                'Avg CI Width', 'Num Players']
    print(position_summary)

    # Elite Shooters (minimal shrinkage)
    
    print(f"\n{'='*70}")
    print("ELITE SHOOTERS - High Volume 3PT (Above the Break)")
    print(f"{'='*70}")
    
    atb3 = df[df['zone'] == 'Above the Break 3'].copy()
    elite = atb3[atb3['attempts'] >= 100].sort_values('posterior_mean', ascending=False).head(10)
    
    print(elite[['player_name', 'position', 'attempts', 'raw_fg_pct', 
                 'posterior_mean', 'ci_lower', 'ci_upper', 'shrinkage']].to_string(index=False))
    
    # Uncertainty Quantification Examples

    print(f"\n{'='*70}")
    print("UNCERTAINTY QUANTIFICATION EXAMPLES")
    print(f"{'='*70}")
    
    # Compare high vs low volume shooters
    print("\nHigh Volume Shooter (narrow CI):")
    high_vol = df.nlargest(1, 'attempts').iloc[0]
    print(f"  {high_vol['player_name']} - {high_vol['zone']}")
    print(f"  Attempts: {high_vol['attempts']}")
    print(f"  Posterior: {high_vol['posterior_mean']:.1%} [{high_vol['ci_lower']:.1%}, {high_vol['ci_upper']:.1%}]")
    print(f"  CI Width: {high_vol['ci_width']:.3f}")
    
    print("\nLow Volume Shooter (wide CI):")
    low_vol = df.nsmallest(1, 'attempts').iloc[0]
    print(f"  {low_vol['player_name']} - {low_vol['zone']}")
    print(f"  Attempts: {low_vol['attempts']}")
    print(f"  Posterior: {low_vol['posterior_mean']:.1%} [{low_vol['ci_lower']:.1%}, {low_vol['ci_upper']:.1%}]")
    print(f"  CI Width: {low_vol['ci_width']:.3f}")
    

    # Biggest Adjustments

    print(f"\n{'='*70}")
    print("BIGGEST BAYESIAN ADJUSTMENTS")
    print(f"{'='*70}")
    
    print("\nMost Regularized DOWN (overperformers on small samples):")
    regularized_down = df.nlargest(5, 'shrinkage')
    print(regularized_down[['player_name', 'zone', 'attempts', 'raw_fg_pct', 
                            'posterior_mean', 'shrinkage']].to_string(index=False))
    
    print("\nMost Regularized UP (underperformers on small samples):")
    regularized_up = df.nsmallest(5, 'shrinkage')
    print(regularized_up[['player_name', 'zone', 'attempts', 'raw_fg_pct', 
                          'posterior_mean', 'shrinkage']].to_string(index=False))



# PLAYER LOOKUP UTILITY

def lookup_player(df, player_name):
    """
    Look up Bayesian estimates for a specific player
    """
    player_data = df[df['player_name'].str.contains(player_name, case=False)]
    
    if len(player_data) == 0:
        print(f"❌ No player found matching '{player_name}'")
        return
    
    print(f"\n{'='*70}")
    print(f"BAYESIAN SHOOTING PROFILE: {player_data.iloc[0]['player_name']}")
    print(f"Position: {player_data.iloc[0]['position']}")
    print(f"{'='*70}\n")
    
    # Sort by attempts
    player_data = player_data.sort_values('attempts', ascending=False)
    
    for _, row in player_data.iterrows():
        print(f"{row['zone']}:")
        print(f"  Sample: {row['makes']}/{row['attempts']} ({row['raw_fg_pct']:.1%})")
        print(f"  Bayesian Est: {row['posterior_mean']:.1%}")
        print(f"  95% CI: [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]")
        print(f"  Prior ({row['position']}): {row['prior_fg_pct']:.1%}")
        print(f"  Shrinkage: {row['shrinkage']:+.3f}")
        print()


# MAIN EXECUTION

def main():
    """Main execution"""
    
    print(f"\n{'#'*70}")
    print(f"# NBA BAYESIAN SHOT ANALYTICS - POSTERIOR COMPUTATION")
    print(f"# Position-Specific Priors with Credible Intervals")
    print(f"{'#'*70}")
    
    # Compute posteriors
    results = compute_all_posteriors()
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    results.to_parquet(OUTPUT_FILE, index=False)
    print(f"Bayesian posteriors saved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024**2:.2f} MB")
    
    # Interactive lookup examples
    print(f"\n{'='*70}")
    print("EXAMPLE PLAYER LOOKUPS")
    print(f"{'='*70}")
    
    lookup_player(results, "Stephen Curry")
    lookup_player(results, "Nikola Jokic")
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}\n")
    print("Key Insights:")
    print("• Position-specific priors provide better comparison baselines")
    print("• High-volume shooters (Curry, Jokic) experience minimal shrinkage")
    print("• Low-volume shooters show substantial regularization toward priors")
    print("• Credible intervals quantify uncertainty (narrow for high volume)")
    print("• Bayesian approach naturally handles sample size differences\n")


if __name__ == "__main__":
    main()