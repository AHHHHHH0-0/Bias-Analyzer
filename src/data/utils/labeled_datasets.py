"""
Fetch and prepare labeled datasets for sentiment and bias classification tasks.
"""
import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def prepare_sentiment_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare sentiment dataset with negative/neutral/positive labels.
    Uses hardcoded examples for demonstration.
    """
    sample_data = [
        ("This movie was absolutely terrible. Waste of time and money.", "negative"),
        ("The plot was boring and the acting was poor. Very disappointed.", "negative"),
        ("I hated every minute of this film. Completely awful.", "negative"),
        ("The service at this restaurant was awful and the food was cold.", "negative"),
        ("This product broke after one day. Complete waste of money.", "negative"),
        ("Climate change policies are destroying our economy completely.", "negative"),
        ("This political candidate is the worst choice for our country.", "negative"),
        ("Healthcare costs are bankrupting families across the nation.", "negative"),
        ("Technology companies are invading our privacy constantly.", "negative"),
        ("The economy is in complete shambles and getting worse daily.", "negative"),
        
        ("The movie was okay, nothing special but not bad either.", "neutral"),
        ("It was an average film with some good moments and some bad ones.", "neutral"),
        ("The movie was decent, though it could have been better.", "neutral"),
        ("The weather today is neither good nor bad, just average.", "neutral"),
        ("The book was interesting but not groundbreaking. Worth a read.", "neutral"),
        ("The climate debate continues with mixed opinions from experts.", "neutral"),
        ("The candidate presented both pros and cons of the policy.", "neutral"),
        ("Healthcare reform requires balancing costs and patient care.", "neutral"),
        ("Artificial intelligence has both benefits and risks to consider.", "neutral"),
        ("The economic outlook remains uncertain with mixed signals.", "neutral"),
        
        ("This was an amazing movie! Loved every second of it.", "positive"),
        ("Fantastic acting and brilliant storyline. Highly recommended!", "positive"),
        ("One of the best films I've ever seen. Absolutely wonderful.", "positive"),
        ("Great customer service and delicious food. Will definitely return!", "positive"),
        ("I love this new phone! The camera quality is incredible.", "positive"),
        ("The new climate initiatives show promising results for the future.", "positive"),
        ("The politician's speech was inspiring and gave me hope.", "positive"),
        ("The new healthcare program is helping millions of Americans.", "positive"),
        ("AI advancements are revolutionizing how we work and live.", "positive"),
        ("Economic indicators show strong growth and job creation.", "positive")
    ]
    
    df = pd.DataFrame(sample_data, columns=["text", "label"])
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_bias_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare political bias dataset with left/center/right labels.
    Uses hardcoded examples for demonstration.
    """
    sample_data = [
        ("The progressive policies will transform America for the better and create equality.", "left"),
        ("The government should expand social programs to help working families.", "left"),
        ("Climate action requires immediate government regulation of all industries.", "left"),
        ("Universal healthcare is a fundamental right that government must provide.", "left"),
        ("Immigration reform should provide pathways to citizenship for all.", "left"),
        ("Gun violence requires immediate action through strict weapon regulations.", "left"),
        ("Wealthy individuals and corporations must pay their fair share in taxes.", "left"),
        ("Climate change demands rapid transition to renewable energy sources.", "left"),
        ("Education funding should prioritize public schools and teacher pay.", "left"),
        ("Labor unions protect workers from corporate exploitation and abuse.", "left"),
        
        ("Environmental policies must balance economic and ecological concerns.", "center"),
        ("Both parties need to work together on comprehensive immigration policy.", "center"),
        ("Gun policy should balance public safety with constitutional rights.", "center"),
        ("Tax policy should be fair and support both growth and public services.", "center"),
        ("Energy policy should diversify sources while protecting the environment.", "center"),
        ("Education policy should support both public and private school options.", "center"),
        ("Workplace policies should balance worker rights with business needs.", "center"),
        ("Budget policy should prioritize essential services while controlling debt.", "center"),
        ("Media outlets should strive for balanced and factual reporting.", "center"),
        ("Immigration policy requires both compassion and border security.", "center"),
        
        ("Corporate tax cuts are essential for economic growth and job creation.", "right"),
        ("Free market solutions are more effective than government intervention.", "right"),
        ("Private healthcare systems deliver better outcomes than government programs.", "right"),
        ("Border security must be strengthened before considering immigration reform.", "right"),
        ("Second Amendment rights must be protected from government overreach.", "right"),
        ("Lower taxes stimulate economic growth and benefit all income levels.", "right"),
        ("Energy independence requires utilizing all domestic energy resources.", "right"),
        ("School choice and competition improve educational outcomes for students.", "right"),
        ("Right-to-work laws protect individual freedom and economic competitiveness.", "right"),
        ("Fiscal responsibility requires reducing government spending and debt.", "right")

    ]
    
    df = pd.DataFrame(sample_data, columns=["text", "label"])
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_labeled_datasets() -> None:
    """Save both sentiment and bias datasets to CSV files."""
    # Create directories if they don't exist
    os.makedirs("src/data/labeled/sentiment", exist_ok=True)
    os.makedirs("src/data/labeled/bias", exist_ok=True)
    
    # Sentiment dataset
    sent_train, sent_test = prepare_sentiment_dataset()
    sent_train.to_csv("src/data/labeled/sentiment/train.csv", index=False)
    sent_test.to_csv("src/data/labeled/sentiment/test.csv", index=False)
    print(f"Sentiment dataset: {len(sent_train)} train, {len(sent_test)} test samples")
    
    # Bias dataset  
    bias_train, bias_test = prepare_bias_dataset()
    bias_train.to_csv("src/data/labeled/bias/train.csv", index=False)
    bias_test.to_csv("src/data/labeled/bias/test.csv", index=False)
    print(f"Bias dataset: {len(bias_train)} train, {len(bias_test)} test samples")
