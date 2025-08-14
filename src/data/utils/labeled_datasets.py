"""
Fetch and prepare labeled datasets for sentiment and bias classification tasks.
"""
import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def prepare_labeled_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    """Save labeled datasets to CSV files."""
    # Create directories if they don't exist
    os.makedirs("src/data/labeled/", exist_ok=True)
    
    # Labeled dataset  
    labeled_train, labeled_test = prepare_labeled_dataset()
    labeled_train.to_csv("src/data/labeled/train.csv", index=False)
    labeled_test.to_csv("src/data/labeled/test.csv", index=False)
    print(f"Labeled dataset: {len(labeled_train)} train, {len(labeled_test)} test samples")
