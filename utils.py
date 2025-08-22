import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
import csv
import io

def export_to_pdf(results, filename=None):
    """
    Export analysis results to PDF format
    Note: This is a placeholder for PDF export functionality
    In a full implementation, you would use libraries like reportlab or weasyprint
    """
    if not filename:
        filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # This would generate a PDF report with:
    # - Summary statistics
    # - Visualizations
    # - Detailed results table
    # - Analysis metadata
    
    return f"PDF export functionality would be implemented here for {filename}"

def create_confusion_matrix():
    """
    Create a sample confusion matrix for demonstration
    In a real implementation, this would use actual vs predicted labels
    """
    # Sample confusion matrix data (would be calculated from real data)
    matrix = np.array([
        [45, 3, 2],   # Actual Positive
        [5, 38, 7],   # Actual Negative  
        [2, 6, 42]    # Actual Neutral
    ])
    
    sentiments = ['Positive', 'Negative', 'Neutral']
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=sentiments,
        y=sentiments,
        colorscale='Blues',
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title="Confusion Matrix (Sample Data)",
        xaxis_title="Predicted Sentiment",
        yaxis_title="Actual Sentiment",
        height=500,
        font=dict(size=12)
    )
    
    # Add accuracy annotations
    total = np.sum(matrix)
    accuracy = np.trace(matrix) / total
    
    fig.add_annotation(
        text=f"Overall Accuracy: {accuracy:.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def calculate_performance_metrics(confusion_matrix):
    """
    Calculate performance metrics from confusion matrix
    """
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square")
    
    n_classes = confusion_matrix.shape[0]
    metrics = {}
    
    # Calculate metrics for each class
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        # Precision, Recall, F1-Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = ['positive', 'negative', 'neutral'][i] if i < 3 else f'class_{i}'
        metrics[f'{class_name}_precision'] = precision
        metrics[f'{class_name}_recall'] = recall
        metrics[f'{class_name}_f1'] = f1
    
    # Overall accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    metrics['accuracy'] = accuracy
    
    # Macro averages
    precisions = [metrics[k] for k in metrics.keys() if 'precision' in k and 'macro' not in k]
    recalls = [metrics[k] for k in metrics.keys() if 'recall' in k and 'macro' not in k]
    f1s = [metrics[k] for k in metrics.keys() if 'f1' in k and 'macro' not in k]
    
    metrics['macro_precision'] = np.mean(precisions) if precisions else 0
    metrics['macro_recall'] = np.mean(recalls) if recalls else 0
    metrics['macro_f1'] = np.mean(f1s) if f1s else 0
    
    return metrics

def format_analysis_summary(results):
    """
    Format analysis results into a readable summary
    """
    if not results:
        return "No analysis results available."
    
    summary = []
    summary.append("=== SENTIMENT ANALYSIS SUMMARY ===\n")
    
    # TextBlob results
    if 'textblob' in results:
        tb = results['textblob']
        summary.append(f"TextBlob Analysis:")
        summary.append(f"  Sentiment: {tb['sentiment'].upper()}")
        summary.append(f"  Polarity: {tb['polarity']:.3f}")
        summary.append(f"  Subjectivity: {tb['subjectivity']:.3f}")
        summary.append(f"  Confidence: {tb['confidence']:.3f}\n")
    
    # VADER results
    if 'vader' in results:
        vader = results['vader']
        summary.append(f"VADER Analysis:")
        summary.append(f"  Sentiment: {vader['sentiment'].upper()}")
        summary.append(f"  Compound Score: {vader['compound']:.3f}")
        summary.append(f"  Positive: {vader['scores']['pos']:.3f}")
        summary.append(f"  Neutral: {vader['scores']['neu']:.3f}")
        summary.append(f"  Negative: {vader['scores']['neg']:.3f}\n")
    
    # Consensus results
    if 'consensus' in results:
        consensus = results['consensus']
        summary.append(f"Consensus Analysis:")
        summary.append(f"  Final Sentiment: {consensus['sentiment'].upper()}")
        summary.append(f"  Confidence: {consensus['confidence']:.3f}")
        summary.append(f"  Agreement Ratio: {consensus['agreement_ratio']:.3f}")
        summary.append(f"  Votes: {consensus['votes']}\n")
    
    # Errors
    errors = [k for k in results.keys() if 'error' in k]
    if errors:
        summary.append("Errors encountered:")
        for error_key in errors:
            summary.append(f"  {error_key}: {results[error_key]}")
    
    return "\n".join(summary)

def validate_api_response(response_data, expected_fields):
    """
    Validate API response data structure
    """
    if not isinstance(response_data, dict):
        return False, "Response must be a dictionary"
    
    missing_fields = []
    for field in expected_fields:
        if field not in response_data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    return True, "Response validation successful"

def sanitize_text_input(text, max_length=10000):
    """
    Sanitize and validate text input
    """
    if not text:
        return "", "Text cannot be empty"
    
    # Convert to string
    text = str(text)
    
    # Remove null bytes and control characters
    text = text.replace('\x00', '')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
        warning = f"Text truncated to {max_length} characters"
    else:
        warning = None
    
    # Basic validation
    if len(text.strip()) < 3:
        return "", "Text too short for meaningful analysis"
    
    return text, warning

def create_export_metadata():
    """
    Create metadata for exported files
    """
    return {
        'export_timestamp': datetime.now().isoformat(),
        'export_format_version': '1.0',
        'application': 'Sentiment Analysis Dashboard',
        'platform': 'Streamlit'
    }

def calculate_text_statistics(text):
    """
    Calculate basic statistics for input text
    """
    if not text:
        return {}
    
    words = text.split()
    sentences = text.split('.')
    
    stats = {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'average_word_length': sum(len(word.strip('.,!?;:"()[]{}')) for word in words) / len(words) if words else 0,
        'average_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0
    }
    
    return stats

def generate_analysis_id():
    """
    Generate unique ID for analysis session
    """
    return f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
