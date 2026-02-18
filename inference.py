"""
Airline Customer Satisfaction Prediction - Inference Script

This script loads the trained model and preprocessing artifacts to make
predictions on new airline review data.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def compute_sentiment_score(text):
    """Compute sentiment score from review text using VADER."""
    if pd.isna(text) or text == '':
        return 0.0
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(str(text))
    return sentiment_dict['compound']


def load_artifacts(models_dir="models"):
    """Load all model artifacts from disk."""
    # Load metadata
    with open(os.path.join(models_dir, "preprocessing_metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model_name = metadata['best_model_name'].lower().replace(' ', '_')
    model = joblib.load(os.path.join(models_dir, f"{model_name}_model.pkl"))
    
    # Load TF-IDF vectorizer
    tfidf_vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    
    # Load ordinal encoder if exists
    encoder_path = os.path.join(models_dir, "ordinal_encoder.pkl")
    ordinal_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    
    return model, tfidf_vectorizer, ordinal_encoder, metadata


def preprocess_input(input_data, tfidf_vectorizer, ordinal_encoder, metadata):
    """Preprocess raw input data to match the model's expected format."""
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Compute sentiment score from review content if not provided
    text_column = metadata['text_column']
    if text_column in df.columns and 'sentiment_score' not in df.columns:
        df['sentiment_score'] = df[text_column].apply(compute_sentiment_score)
    
    # Drop unnecessary columns
    columns_to_drop = [col for col in metadata['columns_to_drop'] if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Extract text
    text_column = metadata['text_column']
    text_data = df[text_column].fillna('') if text_column in df.columns else pd.Series([''])
    
    # Process structured features
    df_structured = df.drop(columns=[text_column], errors='ignore').copy()
    
    # Apply ordinal encoding
    if ordinal_encoder is not None and 'layover_route' in df_structured.columns:
        df_structured[['layover_route']] = ordinal_encoder.transform(df_structured[['layover_route']])
    
    # One-hot encode categorical columns
    categorical_cols = df_structured.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if categorical_cols:
        df_structured = pd.get_dummies(df_structured, columns=categorical_cols, drop_first=True, dtype=int)
    
    df_structured.columns = [col.lower().replace(' ', '_') for col in df_structured.columns]
    df_structured = df_structured.reindex(columns=metadata['structured_feature_names'], fill_value=0)
    
    # Extract TF-IDF features
    tfidf_features = tfidf_vectorizer.transform(text_data)
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out(),
        index=df_structured.index
    )
    
    # Combine features
    final_features = pd.concat([df_structured.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    final_features = final_features.reindex(columns=metadata['final_feature_names'], fill_value=0)
    
    return final_features


def predict(input_data, model, tfidf_vectorizer, ordinal_encoder, metadata):
    """Make prediction on input data."""
    features = preprocess_input(input_data, tfidf_vectorizer, ordinal_encoder, metadata)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return {
        'prediction': 'Satisfied' if prediction == 1 else 'Not Satisfied'
    }


def main():
    """Run inference on example reviews."""
    print("\nAirline Customer Satisfaction Prediction - Inference\n")
    
    # Load artifacts
    model, tfidf_vectorizer, ordinal_encoder, metadata = load_artifacts("models")
    print(f"Model loaded: {metadata['best_model_name']}")
    
    # Example 1: Positive Review
    print("-" * 80)
    print("Example 1: Positive Review")
    positive_review = {
        'review_content': 'Excellent flight with great service. The crew was very friendly and helpful. Food was delicious and seats were comfortable. On time departure and arrival. Highly recommend!',
        'traveller_type': 'Solo Leisure',
        'class': 'Economy Class',
        'verified': 'Trip Verified',
        'layover_route': 'direct_flight'
    }
    
    print ("\nInput Review:")
    print(f"Review Content: {positive_review['review_content']}")
    print(f"Traveller Type: {positive_review['traveller_type']}")
    print(f"Class: {positive_review['class']}")
    print(f"Verified: {positive_review['verified']}")
    print(f"Layover Route: {positive_review['layover_route']}")
    print(f"Sentiment Score: {compute_sentiment_score(positive_review['review_content']):.2f}")

    result = predict(positive_review, model, tfidf_vectorizer, ordinal_encoder, metadata)
    print(f"\nPrediction: {result['prediction']}")

    
    # Example 2: Negative Review
    print("-" * 80)
    print("Example 2: Negative Review")
    negative_review = {
        'review_content': 'Terrible experience. Flight was delayed for hours with no explanation. Staff was rude and unhelpful. Lost my luggage and no one seems to care. Never flying with them again.',
        'traveller_type': 'Business',
        'class': 'Economy Class',
        'verified': 'Trip Verified',
        'layover_route': 'one_layover'
    }
    
    result = predict(negative_review, model, tfidf_vectorizer, ordinal_encoder, metadata)
    print(f"Prediction: {result['prediction']}")
    
    # Example 3: Mixed Review
    print("-" * 80)
    print("Example 3: Mixed Review")
    mixed_review = {
        'review_content': 'The flight was okay. Seats were comfortable but the service was slow. Food was average. Nothing special but got me to my destination.',
        'traveller_type': 'Couple Leisure',
        'class': 'Business Class',
        'verified': 'Not Verified',
        'layover_route': 'direct_flight'
    }
    
    result = predict(mixed_review, model, tfidf_vectorizer, ordinal_encoder, metadata)
    print(f"Prediction: {result['prediction']}")
    
    print("-" * 80)
    
    # take user input
    print("You can also input your own review for prediction.")
    user_review = input("Enter your review: ")
    user_traveller_type = input("Enter traveller type (e.g., Solo Leisure, Business): ")
    user_class = input("Enter class (e.g., Economy Class, Business Class): ")
    user_verified = input("Enter verification status (e.g., Trip Verified, Not Verified): ")
    user_layover_route = input("Enter layover route (e.g., direct_flight, one_layover): ")
    user_sentiment_score = compute_sentiment_score(user_review)
    print(f"Sentiment Score: {user_sentiment_score:.2f}")

    user_input = {
        'review_content': user_review,
        'traveller_type': user_traveller_type,
        'class': user_class,
        'verified': user_verified,
        'layover_route': user_layover_route
    }
    
    result = predict(user_input, model, tfidf_vectorizer, ordinal_encoder, metadata)
    print(f"Prediction: {result['prediction']}")
    print("-" * 80)


if __name__ == "__main__":
    main()
