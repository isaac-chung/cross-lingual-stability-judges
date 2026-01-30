import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _(pd):
    # Read calibration ratings from anonymized CSV file
    import os
    calibration_file = "calibration_query_result_anonymized.csv"
    calibration_df = pd.read_csv(calibration_file)

    len(calibration_df)
    return (calibration_df,)


@app.cell
def _(calibration_df):
    calibration_df['conversation_external_id'].nunique()
    return


@app.cell
def _(pd):
    # Read manual ratings from anonymized CSV file
    manual_file = "manual_query_result_anonymized.csv"
    manual_df = pd.read_csv(manual_file)

    len(manual_df)
    return (manual_df,)


@app.cell
def _(manual_df):
    manual_df['conversation_external_id'].nunique()
    return


@app.cell
def _(manual_df):
    manual_df['reviewer_name'].unique()
    return


@app.cell
def _(calibration_df, manual_df, pd):
    # Standardize column names for union
    calibration_standardized = calibration_df.copy()
    calibration_standardized = calibration_standardized.rename(columns={
        'calibration_reviewer_name': 'reviewer_name',
        'conversation_external_id': 'conversation_id'
    })
    calibration_standardized['data_source'] = 'calibration'

    manual_standardized = manual_df.copy()
    manual_standardized = manual_standardized.rename(columns={
        'conversation_external_id': 'conversation_id'
    })
    manual_standardized['data_source'] = 'manual'

    # Concatenate the dataframes
    combined_raw = pd.concat([calibration_standardized, manual_standardized], ignore_index=True)

    # Check for duplicates before deduplication
    dedup_key = ['reviewer_name', 'conversation_id', 'rating_category_name']
    duplicates_count = combined_raw.duplicated(subset=dedup_key).sum()

    # Deduplicate based on reviewer-conversation-category combination
    # Keep the first occurrence (you can change to 'last' if you prefer manual over calibration)
    combined_df = combined_raw.drop_duplicates(subset=dedup_key, keep='first')
    combined_df = combined_df.sort_values(['conversation_id', 'reviewer_name'])
    return combined_df, combined_raw, dedup_key, duplicates_count


@app.cell
def _(combined_df):
    combined_df['conversation_id'].nunique()
    return


@app.cell
def _(
    calibration_df,
    combined_df,
    combined_raw,
    duplicates_count,
    manual_df,
    mo,
):
    # Display results
    mo.md(f"""
    ### Rating Results Summary

    - **Calibration ratings**: {len(calibration_df)} records
    - **Manual ratings**: {len(manual_df)} records
    - **Combined raw dataset**: {len(combined_raw)} records
    - **Duplicates found**: {duplicates_count} records
    - **Final deduplicated dataset**: {len(combined_df)} records from account_id = 11822

    **Deduplication**: Removed duplicates based on (reviewer_name, conversation_id, rating_category_name) combination, keeping first occurrence.
    """)
    return


@app.cell
def _(combined_raw, dedup_key):
    # Show duplicate analysis
    duplicates_analysis = combined_raw[combined_raw.duplicated(subset=dedup_key, keep=False)]
    duplicates_analysis_sorted = duplicates_analysis.sort_values(dedup_key + ['data_source'])
    duplicates_analysis_sorted
    return


@app.cell
def _(combined_df):
    # Show the final deduplicated dataframe
    combined_df
    return


@app.cell
def _(calibration_df):
    # Show calibration dataframe
    calibration_df.head()
    return


@app.cell
def _(manual_df):
    # Show manual dataframe
    manual_df.head()
    return


@app.cell
def _(combined_df, mo):
    # Show summary statistics for combined data
    mo.md(f"""
    ### Combined Dataset Summary Statistics

    - **Total records**: {len(combined_df)}
    - **Unique reviewers**: {combined_df['reviewer_name'].nunique()}
    - **Unique conversations**: {combined_df['conversation_id'].nunique()}
    - **Unique rating categories**: {combined_df['rating_category_name'].nunique()}
    - **Data sources**: {combined_df['data_source'].value_counts().to_dict()}
    """)
    return


@app.cell
def _(combined_df):
    # Group by reviewer to see distribution across both datasets
    reviewer_stats = combined_df.groupby(['reviewer_name']).agg({
        'conversation_id': 'nunique',
        'rating_category_name': 'nunique',
        'rating_value': ['count', 'mean']
    }).round(2)

    reviewer_stats.columns = ['Conversations', 'Categories', 'Total_Ratings', 'Avg_Rating']
    reviewer_stats
    return


@app.cell
def _(combined_df):
    # Show data source breakdown
    data_source_summary = combined_df.groupby('data_source').agg({
        'reviewer_name': 'nunique',
        'conversation_id': 'nunique',
        'rating_category_name': 'nunique',
        'rating_value': 'count'
    })
    data_source_summary.columns = ['Unique_Reviewers', 'Unique_Conversations', 'Unique_Categories', 'Total_Ratings']
    data_source_summary
    return


@app.cell
def _(combined_df, key_reviewers):
    # Compare ratings. Flag ones that have very drastic differences in fluency ratings. Also flag disagreements in "Does the content make  sense." Use combined_df as a starting point.

    comparison_df = combined_df.pivot_table(
        index=['conversation_id', 'rating_category_name'],
        columns='reviewer_name',
        values='rating_value'
    ).reset_index()

    # Define a function to flag drastic disagreements. 
    def flag_fluency(row):
        """
        Return true if any ratings are 2+ away from each other. 
        """
        if 'fluent' in row['rating_category_name']:
            ratings = row[ key_reviewers ].dropna()
            if len(ratings) >= 2 and ratings.max() - ratings.min() >= 2:
                return True
        return False

    def flag_coherence(row):
        """
        Return true if any ratings are different. 
        """
        if row['rating_category_name'] == 'Does the content make sense?':
            ratings = row[ key_reviewers ].dropna()
            if len(ratings) >= 2 and ratings.nunique() > 1:
                return True
        return False

    comparison_df['drastic_fluency_difference'] = comparison_df.apply(flag_fluency, axis=1)
    comparison_df['coherence_disagreement'] = comparison_df.apply(flag_coherence, axis=1)
    comparison_df.head()
    return (comparison_df,)


@app.cell
def _(comparison_df):
    comparison_df['drastic_fluency_difference'].mean()
    return


@app.cell
def _(comparison_df):
    comparison_df['coherence_disagreement'].mean()
    return


@app.cell
def _(combined_df):
    # get subset with only reviews from key annotators (anonymized)
    key_reviewers = ['Annotator1', 'Annotator2', 'Annotator4']
    key_reviewer_df = combined_df[combined_df['reviewer_name'].isin(key_reviewers)]
    key_reviewer_df.head()
    return key_reviewer_df, key_reviewers


@app.cell
def _(key_reviewer_df):
    # Majority voting
    majority_votes = key_reviewer_df.groupby(['conversation_id', 'rating_category_name']).agg({
        'rating_value': lambda x: x.value_counts().idxmax(),
        'reviewer_name': 'count'
    }).reset_index()
    majority_votes = majority_votes.rename(columns={
        'rating_value': 'majority_rating_value',
        'reviewer_name': 'num_votes'
    })

    ## average over the tickets and group by rating category names. Also report the std.
    majority_votes_agg = majority_votes.groupby('rating_category_name').agg({
        'majority_rating_value': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    majority_votes_agg.columns = ['rating_category_name', 'avg_majority_rating_value', 'std_majority_rating_value']
    majority_votes_agg
    return


@app.cell
def _(key_reviewer_df):
    average_ratings = key_reviewer_df.groupby(['conversation_id', 'rating_category_name']).agg({
        'rating_value': 'mean'
    }).reset_index()
    average_ratings = average_ratings.rename(columns={
        'rating_value': 'average_rating_value'
    })

    ## average over the tickets and group by rating category names.
    average_ratings_agg = average_ratings.groupby('rating_category_name').agg({
        'average_rating_value': ['mean', 'std'],
    }).reset_index()
    average_ratings_agg.columns = ['average_rating_value', 'avg_average_rating_value', 'std_majority_rating_value']
    average_ratings_agg
    return


@app.cell
def _(key_reviewer_df, mo, pd):
    # Import required libraries for Fleiss' kappa
    import numpy as np

    def fleiss_kappa(ratings_matrix):
        """
        Calculate Fleiss' kappa for inter-rater reliability.

        Parameters:
        ratings_matrix: numpy array where rows are subjects (conversations)
                       and columns are rating categories, with values being
                       the count of raters who assigned each rating

        Returns:
        kappa: Fleiss' kappa coefficient
        """
        N, k = ratings_matrix.shape  # N subjects, k categories
        n = ratings_matrix.sum(axis=1)[0]  # number of raters per subject

        # Calculate proportion of ratings for each category
        p_j = ratings_matrix.sum(axis=0) / (N * n)

        # Calculate Pe (expected proportion of agreement)
        Pe = (p_j ** 2).sum()

        # Calculate Po (observed proportion of agreement)
        Po_numerator = 0
        for i in range(N):
            Po_numerator += (ratings_matrix[i] ** 2).sum() - n
        Po = Po_numerator / (N * n * (n - 1))

        # Calculate Fleiss' kappa
        if Pe == 1:
            return 1.0  # Perfect agreement case

        kappa = (Po - Pe) / (1 - Pe)
        return kappa


    def calculate_fleiss_for_category(df, category_name):
        """
        Calculate Fleiss' kappa for a specific rating category.
        """
        # Filter data for the specific category
        category_data = df[df['rating_category_name'] == category_name].copy()

        if len(category_data) == 0:
            return None, "No data for this category"

        # Create a pivot table: conversations x rating_values
        pivot = category_data.pivot_table(
            index='conversation_id',
            columns='rating_value',
            values='reviewer_name',
            aggfunc='count',
            fill_value=0
        )

        # Only include conversations that have ratings from at least 2 reviewers
        conversations_with_multiple_ratings = []
        for conv_id in pivot.index:
            total_ratings = pivot.loc[conv_id].sum()
            if total_ratings >= 2:
                conversations_with_multiple_ratings.append(conv_id)

        if len(conversations_with_multiple_ratings) < 2:
            return None, f"Insufficient data: only {len(conversations_with_multiple_ratings)} conversations with multiple ratings"

        # Filter to conversations with multiple ratings
        filtered_pivot = pivot.loc[conversations_with_multiple_ratings]

        # Convert to numpy array for Fleiss' kappa calculation
        ratings_matrix = filtered_pivot.values

        try:
            kappa = fleiss_kappa(ratings_matrix)
            return kappa, f"Based on {len(conversations_with_multiple_ratings)} conversations"
        except Exception as e:
            return None, f"Error calculating kappa: {str(e)}"

    # Calculate Fleiss' kappa for each rating category
    kappa_results = []

    rating_categories = key_reviewer_df['rating_category_name'].unique()

    for category in rating_categories:
        kappa, note = calculate_fleiss_for_category(key_reviewer_df, category)
        kappa_results.append({
            'rating_category': category,
            'fleiss_kappa': kappa,
            'note': note
        })

    # Create results dataframe
    kappa_df = pd.DataFrame(kappa_results)

    # Display results
    mo.md(f"""
    ### Fleiss' Kappa Inter-Rater Reliability Analysis

    Fleiss' kappa measures agreement between multiple raters on categorical ratings.

    **Interpretation:**
    - κ < 0: Poor agreement (less than chance)
    - 0.0 ≤ κ < 0.20: Slight agreement
    - 0.20 ≤ κ < 0.40: Fair agreement
    - 0.40 ≤ κ < 0.60: Moderate agreement
    - 0.60 ≤ κ < 0.80: Substantial agreement
    - 0.80 ≤ κ ≤ 1.00: Almost perfect agreement

    **Results for Key Reviewers:** {', '.join(key_reviewer_df['reviewer_name'].unique())}
    """)
    return (kappa_df,)


@app.cell
def _(kappa_df):
    ""# Display the Fleiss' kappa results table
    kappa_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
