import streamlit as st
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NBA Bayesian Shot Analytics",
    page_icon="🏀",
    layout="wide"
)

# Configuration
OUTPUT_DIR = Path('data')
POSTERIORS_FILE = OUTPUT_DIR / 'bayesian_posteriors_2023_24.parquet'


@st.cache_data
def load_data():
    df = pd.read_parquet(POSTERIORS_FILE)
    return df


def format_percentage(value):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def format_integer(value):
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def main():
    # Header
    st.title("🏀 NBA Bayesian Shot Analytics")
    st.markdown("**Position-Specific Priors • 2023-24 Season**")
    st.divider()
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {POSTERIORS_FILE}")
        st.info("Run `python bayesian_posteriors.py` first to generate the data.")
        st.stop()
    
    # ========================================================================
    # PLAYER SEARCH
    # ========================================================================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get unique players
        players = df[['player_name', 'position']].drop_duplicates().sort_values('player_name')
        player_names = players['player_name'].tolist()
        
        # Search/select player
        selected_player = st.selectbox(
            "Select Player:",
            player_names,
            index=None,
            placeholder="Choose a player..."
        )
    
    with col2:
        if selected_player:
            player_data = df[df['player_name'] == selected_player]
            position = player_data.iloc[0]['position']
            st.metric("Position", position)
    
    # If no player selected, show instructions
    if not selected_player:
        st.info("👆 Select a player from the dropdown to view their Bayesian shooting statistics")
        st.stop()
    
    st.divider()
    
    # ========================================================================
    # ZONE REFERENCE IMAGE
    # ========================================================================
    st.subheader("📍 NBA Court Zones Reference")
    
    # Display fixed reference image
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    reference_image_path = script_dir / 'court_zones_reference.png'
    
    if reference_image_path.exists():
        # Center the image and make it half size
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(reference_image_path), caption="NBA Shooting Zones", use_column_width=True)
    else:
        st.warning(f"""
        ⚠️ **Court reference image not found**
        
        Looking for: `{reference_image_path}`
        
        Please add a file named `court_zones_reference.png` in the same directory as this script.
        """)
        
        # Fallback: show zone descriptions
        st.info("""
        **NBA Shooting Zones:**
        - **Restricted Area**: Directly under the basket (4-foot arc)
        - **In The Paint (Non-RA)**: Paint area outside restricted zone
        - **Mid-Range**: Between paint and 3-point line
        - **Left Corner 3**: Left corner beyond 3-point line
        - **Right Corner 3**: Right corner beyond 3-point line
        - **Above the Break 3**: Arc area beyond 3-point line
        - **Backcourt**: Beyond half-court
        """)
    
    st.divider()
    
    # ========================================================================
    # STATISTICS TABLE
    # ========================================================================
    st.subheader(f"📊 {selected_player} - Bayesian Shot Statistics")
    
    # Get player data
    player_data = df[df['player_name'] == selected_player].copy()
    
    # Sort by attempts (most common zones first)
    player_data = player_data.sort_values('attempts', ascending=False)
    
    # Create formatted table
    table_data = []
    
    for _, row in player_data.iterrows():
        table_data.append({
            'Zone': row['zone'],
            'Attempts': int(row['attempts']),
            'Makes': int(row['makes']),
            'Raw FG%': format_percentage(row['raw_fg_pct']),
            'Bayesian Est.': format_percentage(row['posterior_mean']),
            '95% CI Lower': format_percentage(row['ci_lower']),
            '95% CI Upper': format_percentage(row['ci_upper']),
            'CI Width': format_percentage(row['ci_width']),
            f'Prior ({row["position"]})': format_percentage(row['prior_fg_pct']),
            'Shrinkage': format_percentage(row['shrinkage']),
        })
    
    # Convert to DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Display with custom styling
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # ========================================================================
    # DETAILED STATISTICS (EXPANDABLE)
    # ========================================================================
    with st.expander("🔬 Advanced Bayesian Parameters"):
        advanced_data = []
        
        for _, row in player_data.iterrows():
            advanced_data.append({
                'Zone': row['zone'],
                'Prior α (alpha)': format_integer(row['prior_alpha']),
                'Prior β (beta)': format_integer(row['prior_beta']),
                'Posterior α': format_integer(row['posterior_alpha']),
                'Posterior β': format_integer(row['posterior_beta']),
                'Sample Size': int(row['attempts']),
                'Effective Sample': int(row['posterior_alpha'] + row['posterior_beta']),
            })
        
        advanced_df = pd.DataFrame(advanced_data)
        st.dataframe(advanced_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    st.divider()
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Best zone
        best_zone = player_data.nlargest(1, 'posterior_mean').iloc[0]
        st.metric(
            "Best Zone",
            best_zone['zone'],
            f"{best_zone['posterior_mean']*100:.1f}%"
        )
    
    with col2:
        # Most attempts
        most_attempts = player_data.nlargest(1, 'attempts').iloc[0]
        st.metric(
            "Most Attempts",
            most_attempts['zone'],
            f"{int(most_attempts['attempts'])} shots"
        )
    
    with col3:
        # Highest confidence (narrowest CI)
        highest_confidence = player_data.nsmallest(1, 'ci_width').iloc[0]
        st.metric(
            "Highest Confidence",
            highest_confidence['zone'],
            f"±{highest_confidence['ci_width']*100/2:.1f}%"
        )
    
    # ========================================================================
    # FOOTER / LEGEND
    # ========================================================================
    st.divider()
    st.markdown("""
    ### 📖 Understanding the Statistics
    
    **Raw FG%**: Simple makes/attempts percentage (can be misleading with small samples)
    
    **Bayesian Estimate**: Adjusted percentage using position-specific priors (more reliable)
    
    **95% Credible Interval**: Range where the true shooting percentage likely falls
    - Narrow interval (< 5%) = high confidence
    - Wide interval (> 15%) = low confidence, need more data
    
    **Shrinkage**: Difference between raw and Bayesian estimate
    - Positive = pulled down (player got lucky on small sample)
    - Negative = pulled up (player got unlucky on small sample)
    - Near zero = large sample, Bayesian agrees with raw percentage
    
    **Prior**: The position-specific baseline (Guards vs Forwards vs Centers shoot differently)
    """)
    
    # Download option
    st.divider()
    
    # Convert to CSV for download
    csv = table_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Statistics as CSV",
        data=csv,
        file_name=f"{selected_player.replace(' ', '_')}_bayesian_stats.csv",
        mime='text/csv'
    )


if __name__ == "__main__":
    main()