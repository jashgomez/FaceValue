import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import os
from PIL import Image
from scipy.stats import pearsonr

# Page configuration
st.set_page_config(
    page_title="CEO Personality Analysis Dashboard",
    page_icon="",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load the CEO dataset"""
    try:
        df = pd.read_csv('clean_data.csv')
        # Clean column names
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Could not find 'clean_data.csv'. Please make sure it's in the same directory as this script.")
        return None

def get_ceo_image(ceo_name, year):
    """
    Find and load CEO image from matched_pictures folder
    Returns PIL Image object or None if not found
    """
    base_dir = "matched_pictures"
    
    # Build the path: matched_pictures/CEO_Name/Year/
    folder_path = os.path.join(base_dir, ceo_name, str(year))
    
    if not os.path.exists(folder_path):
        return None
    
    # Look for the first valid image file
    try:
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, file)
                return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    return None

# Load the data
data = load_data()

if data is None:
    st.stop()

# Create age_group column globally (used in multiple pages)
data['age_group'] = pd.cut(data['scenario.Age'], 
                            bins=[0, 35, 50, 100], 
                            labels=['<35', '35-50', '>50'])

# Title and description
st.title("CEO Personality Analysis Dashboard")
st.markdown("### AI-Generated Personality Inference: Predictive Power, Demographic Trends, and Bias")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis:",
    ["Project Overview", "Results & Future Work", "CEO Lookup", "Bias Explorer", "Predictive Model", "Trends Over Time"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Overview")
st.sidebar.metric("Total Observations", len(data))
st.sidebar.metric("Unique CEOs", data['scenario.CEO'].nunique())
st.sidebar.metric("Companies", data['Company'].nunique())
st.sidebar.metric("Years Covered", f"{data['scenario.Year'].min()}-{data['scenario.Year'].max()}")

# ============================================================
# PAGE 0: PROJECT OVERVIEW
# ============================================================
if page == "Project Overview":
    st.header("Project Overview")
    
    # Introduction
    st.markdown("## Introduction to Project")
    st.markdown("""
    This project investigates whether **AI-generated personality inferences from CEO headshots** 
    have predictive value for firm performance and whether these AI-constructed profiles exhibit 
    systematic demographic bias. With rapid advances in computer vision and large language 
    models, tools that claim to infer personality or leadership style from images are increasingly 
    discussed in contexts such as hiring, executive search, and investment analysis. Yet personality 
    is not directly observable, and AI systems often encode biased correlations present in their 
    training data.
    """)
    
    st.markdown("""
    Using **S&P 500 CEOs between 2010–2020**, the study constructs an original dataset consisting 
    of CEO images, demographic attributes extracted through facial-analysis models, personality 
    descriptors generated through controlled LLM prompts and AI agents, and matched firm-year 
    financial performance. This allows the project to address two core questions:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **1. Predictive Power**
        
        Do LLM-inferred personality traits meaningfully correlate with future firm outcomes 
        such as returns, ROA, or volatility?
        """)
    
    with col2:
        st.info("""
        **2. Bias & Fairness**
        
        Do personality descriptions vary predictably with CEO demographic attributes 
        (age, gender, racial appearance), indicating systematic bias?
        """)
    
    st.markdown("""
    Overall, the project sits at an important intersection of **finance, machine learning, and 
    algorithmic ethics**, providing evidence that informs how AI-based decision tools could impact 
    leadership evaluation, investor behavior, and fairness in corporate governance.
    """)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("## Methodology")
    st.markdown("""
    We built a **CEO-Year / Firm-Year dataset** by merging CEO demographic and personality 
    inferences with annual firm performance. For each S&P 500 firm from 2010–2020, we identify 
    the correct CEO, collect a verified headshot, and use a facial-attribute model to estimate age, 
    gender, and race. These demographic estimates are fed into a fixed LLM prompt to generate a 
    standardized five-trait personality profile for each CEO. We then pull firm performance metrics, 
    returns, ROA, volatility, and market cap, all from public financial sources. After cleaning and 
    aligning all records, we merge the demographic data, LLM-generated traits, and performance 
    outcomes into a single analytic file used for prediction and bias testing.
    """)
    
    st.markdown("---")
    
    # Steps Taken
    st.markdown("## Steps Taken")
    
    # Create a visual flowchart
    steps = [
        ("1️⃣", "Firm-Year Template", "Create S&P 500 firm-year framework (2010-2020)"),
        ("2️⃣", "CEO Identification", "Link each firm-year to serving CEO"),
        ("3️⃣", "Image Collection", "Collect CEO headshots from public sources"),
        ("4️⃣", "Demographic Extraction", "Use DeepFace to extract age, gender, race, emotion"),
        ("5️⃣", "LLM Personality Inference", "Generate personality profiles via AI agents"),
        ("6️⃣", "Financial Data", "Gather firm performance metrics (returns, ROA, etc.)"),
        ("7️⃣", "Data Merging", "Combine all data into unified dataset"),
        ("8️⃣", "Analysis", "Test predictions and examine bias patterns")
    ]
    
    for emoji, title, description in steps:
        col1, col2 = st.columns([0.5, 4])
        with col1:
            st.markdown(f"### {emoji}")
        with col2:
            st.markdown(f"**{title}**")
            st.markdown(description)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("## Dataset Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("### Financial Data")
        st.markdown("""
        - Annual returns
        - ROA & volatility
        - Market capitalization
        - Cumulative performance
        - Industry benchmarks
        """)
    
    with feature_col2:
        st.markdown("### Demographics")
        st.markdown("""
        - Age estimation
        - Gender inference
        - Racial appearance
        - Emotional expression
        - CEO tenure
        """)
    
    with feature_col3:
        st.markdown("### AI Traits")
        st.markdown("""
        - Risk taking (1-5)
        - Leadership strength (1-5)
        - Communication skill (1-5)
        - Financial stewardship (1-5)
        - Overall performance factor
        """)

# ============================================================
# PAGE 0.5: RESULTS & FUTURE WORK
# ============================================================
elif page == "Results & Future Work":
    st.header("Results & Future Work")
    
    # Key Results
    st.markdown("## Key Results")
    
    # Result 1
    st.markdown("### 1. Predictive Power: Limited Evidence")
    
    st.warning("""
    **Finding**: After controlling for firm size, sector, CEO demographics, and fixed effects, 
    investors' projected assessments of CEO traits, including risk-taking, leadership, communication, 
    and financial stewardship, **do not significantly predict next-year firm returns**.
    """)
    
    st.markdown("""
    This suggests that while investor judgments may be influenced by observable traits, these 
    perceptions are **not reliable indicators of firm performance**, highlighting the presence 
    of potential bias in expectations.
    """)
    
    st.markdown("---")
    
    # Result 2
    st.markdown("### 2. Systematic Demographic Bias")
    
    st.error("""
    **Finding**: Our results show **consistent demographic patterns** in how the model assigns 
    personality traits, indicating systematic bias in AI-generated assessments.
    """)
    
    # Gender Bias
    st.markdown("Gender-Based Differences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Male CEOs:**
        - Slightly higher risk-taking scores
        - More "assertive" descriptors
        """)
    
    with col2:
        st.markdown("""
        **Female CEOs:**
        - Higher communication scores
        - Higher financial stewardship scores
        - Gender differences relatively small
        """)
    
    # Age Bias
    st.markdown("Age-Based Patterns (Strongest Effect)")
    
    age_col1, age_col2, age_col3 = st.columns(3)
    
    with age_col1:
        st.success("""
        **Younger CEOs (<35)**
        - Highest risk-taking scores
        - "Bold" and "aggressive"
        """)
    
    with age_col2:
        st.info("""
        **Middle Age (35-50)**
        - Moderate risk-taking
        - Balanced profiles
        """)
    
    with age_col3:
        st.warning("""
        **Older CEOs (>50)**
        - Lowest risk-taking scores
        - "Conservative" descriptors
        """)
    
    st.markdown("""
    The heatmap analysis shows a **clear decline in perceived boldness as age increases**, 
    with the lowest scores appearing in the group older than fifty. This suggests that the model 
    links youth with assertiveness and portrays older CEOs as more conservative.
    """)
    
    # Race Bias
    st.markdown("Race-Based Differences (Larger than Gender)")
    
    race_results = pd.DataFrame({
        'Demographic Group': ['Black CEOs', 'Indian CEOs', 'Latino/Hispanic CEOs', 'White CEOs'],
        'Key Finding': [
            'Lowest risk-taking scores',
            'Highest communication & financial stewardship',
            'Lowest communication scores',
            'Consistently near middle across all traits'
        ],
        'Implication': [
            'Stereotyped as conservative',
            'Stereotyped as technically competent',
            'Communication bias',
            'Default "neutral" baseline'
        ]
    })
    
    st.dataframe(race_results, use_container_width=True, hide_index=True)
    
    st.markdown("""
    These combined patterns indicate that **the AI is using demographic cues when shaping 
    personality judgments**, suggesting systematic bias in how leadership qualities are inferred 
    across groups.
    """)
    
    st.markdown("---")
    
    # Future Development
    st.markdown("Future Development")
    
    st.markdown("""
    With additional time and resources, this project could be expanded in several meaningful ways:
    """)
    
    # Expansion Areas
    expansion_col1, expansion_col2 = st.columns(2)
    
    with expansion_col1:
        st.markdown("Data Expansion")
        st.markdown("""
        - **Scale beyond S&P 500**
          - International firms
          - Private companies
          - Longer historical window
          - Increased statistical power
        
        - **Improved Image Quality**
          - Corporate filing photos
          - Professional headshot databases
          - Standardized image sources
          - Better facial attribute accuracy
        """)
        
        st.markdown("Technical Improvements")
        st.markdown("""
        - **Advanced Computer Vision**
          - Facial structure analysis
          - Expression recognition
          - Less reliance on demographics
          - Multi-modal analysis
        
        - **Enhanced Financial Variables**
          - ESG indicators
          - Analyst sentiment
          - Earnings call transcripts
          - Market sentiment analysis
        """)
    
    with expansion_col2:
        st.markdown("Platform Development")
        st.markdown("""
        - **Interactive Dashboard**
          - Real-time predictions
          - User image uploads
          - Live bias analysis
          - Automated reporting
        
        - **Web Platform**
          - Public API access
          - Custom model training
          - Comparative analysis tools
          - Educational resources
        """)
        
        st.markdown("Research Extensions")
        st.markdown("""
        - **Longitudinal Studies**
          - CEO career trajectories
          - Performance over tenure
          - Trait stability analysis
        
        - **Intervention Testing**
          - Bias correction methods
          - Debiasing algorithms
          - Fairness-aware models
        """)
    
    st.markdown("---")
    
    # Implications
    st.markdown("Key Implications")
    
    implications = [
        ("AI-based assessment tools encode systematic demographic biases"),
        ("Appearance-based personality inference has limited predictive value"),
        ("Transparency and bias testing are essential for AI decision tools"),
        ("Fairness concerns apply to hiring, investing, and leadership evaluation"),
        ("Regulations may be needed to govern AI use in high-stakes decisions")
    ]
    
    for text in implications:
        st.markdown(f"**{text}")

# ============================================================
# PAGE 1: CEO LOOKUP
# ============================================================
elif page == "CEO Lookup":
    st.header("CEO Lookup & Individual Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select CEO")
        
        # Company selector
        companies = sorted(data['Company'].unique())
        selected_company = st.selectbox("Select Company:", companies)
        
        # Filter CEOs by company
        company_data = data[data['Company'] == selected_company]
        ceos = sorted(company_data['scenario.CEO'].unique())
        selected_ceo = st.selectbox("Select CEO:", ceos)
        
        # Year selector
        ceo_years = sorted(company_data[company_data['scenario.CEO'] == selected_ceo]['scenario.Year'].unique())
        selected_year = st.selectbox("Select Year:", ceo_years)
        
        # Get the specific record
        ceo_record = data[(data['scenario.CEO'] == selected_ceo) & 
                         (data['scenario.Year'] == selected_year)].iloc[0]
        
        # Display CEO Image
        st.markdown("---")
        st.subheader("CEO Photo")
        ceo_image = get_ceo_image(selected_ceo, selected_year)
        
        if ceo_image:
            st.image(ceo_image, caption=f"{selected_ceo} ({selected_year})", use_container_width=True)
        else:
            st.info("📷 No image available for this CEO-Year combination")
        
        st.markdown("---")
        st.subheader("Demographic Profile")
        
        # Demographic information
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            st.metric("Age", int(ceo_record['scenario.Age']))
            st.metric("Gender", ceo_record['scenario.dominant_gender'])
        with demo_col2:
            st.metric("Race", ceo_record['scenario.dominant_race'].title())
            st.metric("Emotion", ceo_record['scenario.dominant_emotion'].title())
    
    with col2:
        st.subheader("Performance Analysis")
        
        # Performance metrics
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("Daily Return", f"{ceo_record['Return']:.4f}")
            st.metric("Year Cumulative Return", f"{ceo_record['Year_Cum_Ret_Overall']:.4f}")
        with perf_col2:
            st.metric("Tenure Cumulative Return", f"{ceo_record['Tenure_Cum_Ret_Overall']:.4f}")
            st.metric("Expected Performance", f"{ceo_record['expected_performance_factor']:.2f}")
        with perf_col3:
            st.metric("Industry", ceo_record['Industry'])
            st.metric("Sector", ceo_record['Sector'])
        
        st.markdown("---")
        
        # AI-Generated Personality Scores
        st.subheader("AI-Generated Personality Profile")
        
        personality_traits = {
            'Risk Taking': ceo_record['answer.Projected_risk_taking'],
            'Leadership': ceo_record['answer.Projected_leadership_strength'],
            'Communication': ceo_record['answer.Projected_communication_skill'],
            'Financial Stewardship': ceo_record['answer.Projected_financial_stewardship']
        }
        
        # Radar chart for personality
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(personality_traits.values()),
            theta=list(personality_traits.keys()),
            fill='toself',
            name='Personality Profile',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='rgb(59, 130, 246)', width=2)
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Scores - aligned with Demographic Profile
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Detailed Scores")
        
    with col_right:
        score_cols = st.columns(4)
        for idx, (trait, score) in enumerate(personality_traits.items()):
            with score_cols[idx]:
                st.metric(trait, f"{score:.1f}/5")
    
    # CEO Timeline
    st.markdown("---")
    st.subheader("CEO Performance Over Time")
    
    ceo_timeline = company_data[company_data['scenario.CEO'] == selected_ceo].sort_values('scenario.Year')
    
    if len(ceo_timeline) > 1:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Cumulative Returns", "AI Expected Performance Factor"),
            vertical_spacing=0.15
        )
        
        # Returns over time
        fig.add_trace(
            go.Scatter(x=ceo_timeline['scenario.Year'], 
                      y=ceo_timeline['Tenure_Cum_Ret_Overall'],
                      mode='lines+markers',
                      name='Tenure Cumulative Return',
                      line=dict(color='#3b82f6', width=3)),
            row=1, col=1
        )
        
        # Expected performance over time
        fig.add_trace(
            go.Scatter(x=ceo_timeline['scenario.Year'], 
                      y=ceo_timeline['expected_performance_factor'],
                      mode='lines+markers',
                      name='Expected Performance',
                      line=dict(color='#10b981', width=3)),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Score (1-5)", row=2, col=1)
        fig.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Only one year of data available for this CEO.")

# ============================================================
# PAGE 2: BIAS EXPLORER
# ============================================================
elif page == "Bias Explorer":
    st.header("Bias Explorer: Demographic Disparities in AI Ratings")
    
    st.warning("""
    **⚠️ Bias Detection Alert**: This analysis reveals systematic differences in AI-generated trait 
    assignments across demographic groups. These patterns suggest potential algorithmic bias.
    """)
    
    # Demographic Group Analysis
    st.subheader("Personality Traits by Demographic Groups")
    
    # Select dimension for analysis
    dimension = st.selectbox(
        "Select Demographic Dimension:",
        ["Gender", "Race", "Age Group", "Emotion"]
    )
    
    # Map selection to column name
    dimension_map = {
        "Gender": "scenario.dominant_gender",
        "Race": "scenario.dominant_race",
        "Age Group": "age_group",
        "Emotion": "scenario.dominant_emotion"
    }
    
    selected_col = dimension_map[dimension]
    
    # Aggregate by demographic
    traits = ['answer.Projected_risk_taking', 'answer.Projected_leadership_strength', 
              'answer.Projected_communication_skill', 'answer.Projected_financial_stewardship']
    
    agg_data = data.groupby(selected_col)[traits].mean().reset_index()
    
    # Rename for display
    agg_data.columns = [dimension, 'Risk Taking', 'Leadership', 'Communication', 'Financial Stewardship']
    
    # Create grouped bar chart
    fig = go.Figure()
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    for idx, trait in enumerate(['Risk Taking', 'Leadership', 'Communication', 'Financial Stewardship']):
        fig.add_trace(go.Bar(
            name=trait,
            x=agg_data[dimension],
            y=agg_data[trait],
            marker_color=colors[idx]
        ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title=dimension,
        yaxis_title="Average Score (1-5)",
        height=500,
        yaxis_range=[0, 5]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Summary Table
    st.subheader("📋 Statistical Summary by Group")
    
    summary_table = agg_data.round(2)
    st.dataframe(summary_table, use_container_width=True)
    
    # Heatmap for multiple dimensions
    st.markdown("---")
    st.subheader("Bias Heatmap: Gender × Age Group")
    
    heatmap_data = data.groupby(['scenario.dominant_gender', 'age_group'])[
        'answer.Projected_risk_taking'].mean().reset_index()
    
    heatmap_pivot = heatmap_data.pivot(
        index='scenario.dominant_gender', 
        columns='age_group', 
        values='answer.Projected_risk_taking'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='RdYlBu',
        text=heatmap_pivot.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Risk Taking Score")
    ))
    
    fig.update_layout(
        title="Average Risk Taking Score by Gender and Age Group",
        xaxis_title="Age Group",
        yaxis_title="Gender",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Findings
    st.markdown("---")
    st.subheader("Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Observed Patterns:**")
        
        # Calculate some statistics
        male_risk = data[data['scenario.dominant_gender'] == 'Man']['answer.Projected_risk_taking'].mean()
        female_risk = data[data['scenario.dominant_gender'] == 'Woman']['answer.Projected_risk_taking'].mean()
        
        young_risk = data[data['age_group'] == '<35']['answer.Projected_risk_taking'].mean()
        old_risk = data[data['age_group'] == '>50']['answer.Projected_risk_taking'].mean()
        
        st.markdown(f"- Male CEOs: Avg Risk Score = **{male_risk:.2f}**")
        st.markdown(f"- Female CEOs: Avg Risk Score = **{female_risk:.2f}**")
        st.markdown(f"- Gender Gap: **{abs(male_risk - female_risk):.2f} points**")
        st.markdown(f"- Young (<35): Avg Risk Score = **{young_risk:.2f}**")
        st.markdown(f"- Older (>50): Avg Risk Score = **{old_risk:.2f}**")
        st.markdown(f"- Age Gap: **{abs(young_risk - old_risk):.2f} points**")
    
    with col2:
        st.markdown("**Implications:**")
        st.markdown("- Systematic bias in AI trait inference")
        st.markdown("- Stereotypical associations (age ↔ risk)")
        st.markdown("- Gender-based differentiation in ratings")
        st.markdown("- Emotion may influence perceived competence")
        st.markdown("- Need for bias correction mechanisms")

# ============================================================
# PAGE 3: PREDICTIVE MODEL
# ============================================================
elif page == "Predictive Model":
    st.header("Predictive Model: Can AI Traits Predict Performance?")
    
    # Overall model metrics
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate correlations
    from scipy.stats import pearsonr
    
    traits = ['answer.Projected_risk_taking', 'answer.Projected_leadership_strength', 
              'answer.Projected_communication_skill', 'answer.Projected_financial_stewardship',
              'expected_performance_factor']
    
    performance_metric = 'Tenure_Cum_Ret_Overall'
    
    # Clean data for correlation
    clean_data = data[traits + [performance_metric]].dropna()
    
    # Calculate R² for expected performance factor
    corr, _ = pearsonr(clean_data['expected_performance_factor'], clean_data[performance_metric])
    r_squared = corr ** 2
    
    with col1:
        st.metric("R² Score", f"{r_squared:.3f}", help="Variance explained by the model")
    with col2:
        st.metric("Sample Size", f"{len(clean_data):,}", help="Valid observations")
    with col3:
        st.metric("Correlation", f"{corr:.3f}", help="Pearson correlation coefficient")
    with col4:
        mean_return = clean_data[performance_metric].mean()
        st.metric("Mean Return", f"{mean_return:.3f}", help="Average cumulative return")
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("Trait-Performance Correlation")
    
    correlations = {}
    for trait in traits:
        corr_val, _ = pearsonr(clean_data[trait], clean_data[performance_metric])
        trait_name = trait.replace('answer.Projected_', '').replace('_', ' ').title()
        if trait == 'expected_performance_factor':
            trait_name = 'Expected Performance (Overall)'
        correlations[trait_name] = corr_val
    
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Trait', 'Correlation'])
    corr_df = corr_df.sort_values('Correlation', ascending=True)
    
    fig = px.bar(
        corr_df,
        y='Trait',
        x='Correlation',
        orientation='h',
        color='Correlation',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    fig.update_layout(
        height=400,
        xaxis_title="Pearson Correlation with Tenure Cumulative Return",
        yaxis_title="",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter Plot: Predicted vs Actual
    st.markdown("---")
    st.subheader("Expected Performance vs. Actual Returns")
    
    # Select which trait to visualize
    trait_option = st.selectbox(
        "Select Trait for Visualization:",
        ['Expected Performance (Overall)', 'Risk Taking', 'Leadership', 
         'Communication', 'Financial Stewardship']
    )
    
    trait_map = {
        'Expected Performance (Overall)': 'expected_performance_factor',
        'Risk Taking': 'answer.Projected_risk_taking',
        'Leadership': 'answer.Projected_leadership_strength',
        'Communication': 'answer.Projected_communication_skill',
        'Financial Stewardship': 'answer.Projected_financial_stewardship'
    }
    
    selected_trait = trait_map[trait_option]
    
    fig = px.scatter(
        clean_data,
        x=selected_trait,
        y=performance_metric,
        opacity=0.6,
        color=selected_trait,
        color_continuous_scale='Viridis',
        labels={
            selected_trait: trait_option + ' Score',
            performance_metric: 'Tenure Cumulative Return'
        },
        trendline="ols"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by demographic groups
    st.markdown("---")
    st.subheader("Performance by Demographic Group")
    
    group_col1, group_col2 = st.columns(2)
    
    with group_col1:
        # By Gender
        gender_perf = data.groupby('scenario.dominant_gender').agg({
            performance_metric: 'mean',
            'expected_performance_factor': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Actual Return',
            x=gender_perf['scenario.dominant_gender'],
            y=gender_perf[performance_metric],
            marker_color='#3b82f6'
        ))
        fig.add_trace(go.Bar(
            name='Expected Performance',
            x=gender_perf['scenario.dominant_gender'],
            y=gender_perf['expected_performance_factor'],
            marker_color='#10b981'
        ))
        fig.update_layout(
            title="Performance by Gender",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with group_col2:
        # By Age Group
        age_perf = data.groupby('age_group').agg({
            performance_metric: 'mean',
            'expected_performance_factor': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Actual Return',
            x=age_perf['age_group'],
            y=age_perf[performance_metric],
            marker_color='#f59e0b'
        ))
        fig.add_trace(go.Bar(
            name='Expected Performance',
            x=age_perf['age_group'],
            y=age_perf['expected_performance_factor'],
            marker_color='#8b5cf6'
        ))
        fig.update_layout(
            title="Performance by Age Group",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Insights
    st.markdown("---")
    st.subheader("Key Insights")
    
    with st.expander("View Detailed Analysis"):
        st.markdown(f"""
        **Model Interpretation:**
        
        1. **Overall Predictive Power**: The model explains **{r_squared*100:.1f}%** of variance in firm performance
        2. **Correlation Strength**: {trait_option} shows a correlation of **{corr:.3f}** with actual returns
        3. **Practical Significance**: {"Weak" if abs(corr) < 0.3 else "Moderate" if abs(corr) < 0.6 else "Strong"} relationship between AI ratings and actual performance
        
        **Limitations:**
        - AI-generated traits capture only limited aspects of CEO effectiveness
        - Many other factors influence firm performance (market conditions, industry trends, etc.)
        - Correlation does not imply causation
        - Potential for algorithmic bias affecting predictions
        
        **Recommendations:**
        - Use as supplementary tool, not primary decision factor
        - Always combine with traditional performance metrics
        - Be aware of demographic biases in AI ratings
        - Validate findings with larger, more diverse datasets
        """)

# ============================================================
# PAGE 4: TRENDS OVER TIME
# ============================================================
elif page == "Trends Over Time":
    st.header("Temporal Trends: Evolution of AI Ratings (2010-2019)")
    
    st.info("This analysis shows how AI personality predictions and their correlation with performance evolved over the decade.")
    
    # Aggregate by year
    yearly_data = data.groupby('scenario.Year').agg({
        'answer.Projected_risk_taking': 'mean',
        'answer.Projected_leadership_strength': 'mean',
        'answer.Projected_communication_skill': 'mean',
        'answer.Projected_financial_stewardship': 'mean',
        'expected_performance_factor': 'mean',
        'Tenure_Cum_Ret_Overall': 'mean',
        'scenario.CEO': 'count'
    }).reset_index()
    
    yearly_data.columns = ['Year', 'Risk Taking', 'Leadership', 'Communication', 
                          'Financial Stewardship', 'Expected Performance', 
                          'Avg Return', 'Count']
    
    # Evolution of traits
    st.subheader("Evolution of Average Personality Traits")
    
    fig = go.Figure()
    
    traits_to_plot = ['Risk Taking', 'Leadership', 'Communication', 'Financial Stewardship']
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    
    for trait, color in zip(traits_to_plot, colors):
        fig.add_trace(go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data[trait],
            mode='lines+markers',
            name=trait,
            line=dict(width=3, color=color),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Score (1-5)",
        height=500,
        hovermode='x unified',
        yaxis_range=[0, 5]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Expected vs Actual Performance Over Time
    st.markdown("---")
    st.subheader("Expected Performance vs. Actual Returns Over Time")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Expected Performance'],
            name="AI Expected Performance",
            line=dict(color='#10b981', width=3),
            mode='lines+markers'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Avg Return'],
            name="Actual Cumulative Return",
            line=dict(color='#3b82f6', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Expected Performance (1-5)", secondary_y=False)
    fig.update_yaxes(title_text="Average Cumulative Return", secondary_y=True)
    fig.update_layout(height=500, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample size over time
    st.markdown("---")
    st.subheader("Data Coverage Over Time")
    
    fig = px.bar(
        yearly_data,
        x='Year',
        y='Count',
        labels={'Count': 'Number of CEO-Year Observations'},
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation trends
    st.markdown("---")
    st.subheader("Year-by-Year Correlation Analysis")
    
    # Calculate correlation for each year
    correlation_by_year = []
    for year in sorted(data['scenario.Year'].unique()):
        year_data = data[data['scenario.Year'] == year][
            ['expected_performance_factor', 'Tenure_Cum_Ret_Overall']
        ].dropna()
        
        if len(year_data) > 5:  # Need at least some data points
            corr, _ = pearsonr(year_data['expected_performance_factor'], 
                             year_data['Tenure_Cum_Ret_Overall'])
            correlation_by_year.append({'Year': year, 'Correlation': corr})
    
    corr_df = pd.DataFrame(correlation_by_year)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr_df['Year'],
        y=corr_df['Correlation'],
        mode='lines+markers',
        name='Correlation',
        line=dict(width=3, color='#8b5cf6'),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.2)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="No Correlation")
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Correlation Coefficient",
        height=400,
        yaxis_range=[-1, 1]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.markdown("---")
    st.subheader("Key Temporal Insights")
    
    with st.expander("View Detailed Findings"):
        risk_change = yearly_data['Risk Taking'].iloc[-1] - yearly_data['Risk Taking'].iloc[0]
        leadership_change = yearly_data['Leadership'].iloc[-1] - yearly_data['Leadership'].iloc[0]
        
        st.markdown(f"""
        **Observed Trends (2010-2019):**
        
        1. **Risk Taking Evolution**: {"Increased" if risk_change > 0 else "Decreased"} by **{abs(risk_change):.2f} points** over the decade
        2. **Leadership Ratings**: {"Increased" if leadership_change > 0 else "Decreased"} by **{abs(leadership_change):.2f} points**
        3. **Data Coverage**: Year **{yearly_data.loc[yearly_data['Count'].idxmax(), 'Year']:.0f}** has the most observations (**{yearly_data['Count'].max():.0f}** CEO-years)
        4. **Correlation Stability**: {"Relatively stable" if corr_df['Correlation'].std() < 0.2 else "Variable"} correlation over time (std: **{corr_df['Correlation'].std():.3f}**)
        
        **Implications:**
        - AI ratings may reflect changing societal perceptions of leadership
        - Temporal trends suggest evolving biases or standards
        - Correlation strength varies by market conditions and time period
        - Longitudinal validation is essential for AI-based assessment tools
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>FIN 377 Final Project</strong> | Cecy Benitez & Jashlyn Gomez</p>
    <p><em>AI-Generated Personality Inference of CEOs: Predictive Power, Demographic Trends, and Bias</em></p>
    <p style='font-size: 0.9em;'>Dataset: {dataset_years} | {total_obs} observations across {num_companies} companies</p>
</div>
""".format(
    dataset_years=f"{data['scenario.Year'].min()}-{data['scenario.Year'].max()}",
    total_obs=len(data),
    num_companies=data['Company'].nunique()
), unsafe_allow_html=True)