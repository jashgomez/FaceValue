# FaceValue
AI-Generated Personality Inference of CEOs: Predictive Power, Demographic Trends, and Bias

By Cecy Benitez and Jashlyn Gomez

## Introduction 
This project investigates whether AI-generated personality inferences from CEO headshots
have predictive value for firm performance and whether these AI-constructed profiles exhibit
systematic demographic bias. With rapid advances in computer vision and large language
models, tools that claim to infer personality or leadership style from images are increasingly
discussed in contexts such as hiring, executive search, and investment analysis. Yet personality
is not directly observable, and AI systems often encode biased correlations present in their
training data.
Using S&P 500 CEOs between 2010–2020, the study constructs an original dataset consisting
of CEO images, demographic attributes extracted through facial-analysis models, personality
descriptors generated through controlled LLM prompts and AI agents, and matched firm-year
financial performance. This allows the project to address two core questions:
1. Predictive Power -– Do LLM-inferred personality traits meaningfully correlate with future
firm outcomes such as returns, ROA, or volatility?
2. Bias & Fairness — Do personality descriptions vary predictably with CEO demographic
attributes (age, gender, racial appearance), indicating systematic bias?
Overall, the project sits at an important intersection of finance, machine learning, and
algorithmic ethics, providing evidence that informs how AI-based decision tools could impact
leadership evaluation, investor behavior, and fairness in corporate governance.

## Methodology
We built a CEO-Year / Firm-Year dataset by merging CEO demographic and personality
inferences with annual firm performance. For each S&P 500 firm from 2010–2020, we identify
the correct CEO, collect a verified headshot, and use a facial-attribute model to estimate age,
gender, and race. These demographic estimates are fed into a fixed LLM prompt to generate a
standardized five-trait personality profile for each CEO. We then pull firm performance metrics,
returns, ROA, volatility, and market cap, all from public financial sources. After cleaning and
aligning all records, we merge the demographic data, LLM-generated traits, and performance
outcomes into a single analytic file used for prediction and bias testing.

## Results 
1. Investors’ projected assessments of CEO traits do not significantly predict next-year firm returns after accounting for firm size, sector, demographics, and fixed effects, suggesting these judgments reflect bias rather than true performance signals.
2. The model also shows clear demographic patterns in how it assigns traits, with gender, age, and race influencing inferred qualities in systematic ways that indicate the AI relies on demographic cues when forming personality judgments.
