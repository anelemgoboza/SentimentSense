import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class SentimentVisualizer:
    """
    Visualization utilities for sentiment analysis results
    """
    
    def __init__(self):
        self.color_map = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4'    # Steel Blue
        }
    
    def create_sentiment_pie_chart(self, vader_scores):
        """
        Create pie chart from VADER sentiment scores
        """
        labels = ['Positive', 'Neutral', 'Negative']
        values = [vader_scores['pos'], vader_scores['neu'], vader_scores['neg']]
        colors = [self.color_map['positive'], self.color_map['neutral'], self.color_map['negative']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="VADER Sentiment Score Distribution",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_sentiment_distribution_chart(self, sentiment_counts):
        """
        Create bar chart of sentiment distribution
        """
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        colors = [self.color_map.get(sentiment, '#808080') for sentiment in sentiments]
        
        fig = go.Figure(data=[go.Bar(
            x=sentiments,
            y=counts,
            marker=dict(color=colors),
            text=counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_confidence_histogram(self, confidence_scores, bins=20):
        """
        Create histogram of confidence scores
        """
        fig = px.histogram(
            x=confidence_scores,
            nbins=bins,
            title="Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Frequency'}
        )
        
        fig.update_layout(height=400)
        fig.update_traces(marker_color='skyblue', opacity=0.7)
        
        return fig
    
    def create_sentiment_timeline(self, timeline_data):
        """
        Create timeline visualization of sentiment analysis
        """
        if not timeline_data:
            return go.Figure()
        
        df = pd.DataFrame(timeline_data)
        
        fig = px.scatter(
            df,
            x='timestamp',
            y='confidence',
            color='sentiment',
            size='confidence',
            color_discrete_map=self.color_map,
            title="Sentiment Analysis Timeline",
            labels={'confidence': 'Confidence Score', 'timestamp': 'Time'}
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_comparison_chart(self, textblob_results, vader_results):
        """
        Create comparison chart between TextBlob and VADER results
        """
        # Prepare data for comparison
        sentiments = ['Positive', 'Negative', 'Neutral']
        
        textblob_counts = [0, 0, 0]
        vader_counts = [0, 0, 0]
        
        sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        for result in textblob_results:
            idx = sentiment_map.get(result, 2)
            textblob_counts[idx] += 1
        
        for result in vader_results:
            idx = sentiment_map.get(result, 2)
            vader_counts[idx] += 1
        
        fig = go.Figure(data=[
            go.Bar(name='TextBlob', x=sentiments, y=textblob_counts, marker_color='lightblue'),
            go.Bar(name='VADER', x=sentiments, y=vader_counts, marker_color='lightcoral')
        ])
        
        fig.update_layout(
            barmode='group',
            title="TextBlob vs VADER Sentiment Analysis Comparison",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_accuracy_metrics_chart(self, accuracy_data):
        """
        Create chart showing accuracy metrics
        """
        metrics = list(accuracy_data.keys())
        values = list(accuracy_data.values())
        
        fig = go.Figure(data=[go.Bar(
            x=metrics,
            y=values,
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"{v:.3f}" for v in values],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Accuracy Metrics",
            xaxis_title="Metric",
            yaxis_title="Score",
            height=400
        )
        
        return fig
    
    def create_word_frequency_chart(self, word_frequencies, top_n=20):
        """
        Create horizontal bar chart of word frequencies
        """
        if not word_frequencies:
            return go.Figure()
        
        # Sort and get top N words
        sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, frequencies = zip(*sorted_words) if sorted_words else ([], [])
        
        fig = go.Figure(data=[go.Bar(
            x=list(frequencies),
            y=list(words),
            orientation='h',
            marker=dict(color='lightgreen', opacity=0.7)
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Most Frequent Words",
            xaxis_title="Frequency",
            yaxis_title="Words",
            height=max(400, len(words) * 25),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def create_sentiment_heatmap(self, sentiment_matrix):
        """
        Create heatmap for sentiment analysis confusion matrix
        """
        sentiments = ['Positive', 'Negative', 'Neutral']
        
        fig = go.Figure(data=go.Heatmap(
            z=sentiment_matrix,
            x=sentiments,
            y=sentiments,
            colorscale='RdYlBu_r',
            text=sentiment_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
            title="Sentiment Analysis Confusion Matrix",
            xaxis_title="Predicted Sentiment",
            yaxis_title="Actual Sentiment",
            height=500
        )
        
        return fig
    
    def create_batch_summary_dashboard(self, batch_results):
        """
        Create comprehensive dashboard for batch analysis results
        """
        # Calculate statistics
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        confidence_scores = []
        text_lengths = []
        
        for result in batch_results:
            if 'results' in result and 'consensus' in result['results']:
                consensus = result['results']['consensus']
                sentiment = consensus.get('sentiment', 'neutral')
                confidence = consensus.get('confidence', 0)
                
                sentiment_counts[sentiment] += 1
                confidence_scores.append(confidence)
                text_lengths.append(len(result.get('text', '')))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Confidence Scores', 'Text Length Distribution', 'Sentiment vs Confidence'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Sentiment distribution pie chart
        fig.add_trace(go.Pie(
            labels=list(sentiment_counts.keys()),
            values=list(sentiment_counts.values()),
            marker=dict(colors=[self.color_map[s] for s in sentiment_counts.keys()]),
            name="Sentiment"
        ), row=1, col=1)
        
        # Confidence histogram
        fig.add_trace(go.Histogram(
            x=confidence_scores,
            name="Confidence",
            marker_color='skyblue',
            opacity=0.7
        ), row=1, col=2)
        
        # Text length histogram
        fig.add_trace(go.Histogram(
            x=text_lengths,
            name="Text Length",
            marker_color='lightgreen',
            opacity=0.7
        ), row=2, col=1)
        
        # Sentiment vs Confidence scatter
        sentiment_labels = []
        for result in batch_results:
            if 'results' in result and 'consensus' in result['results']:
                sentiment_labels.append(result['results']['consensus'].get('sentiment', 'neutral'))
            else:
                sentiment_labels.append('neutral')
        
        fig.add_trace(go.Scatter(
            x=confidence_scores,
            y=[sentiment_counts[s] for s in sentiment_labels[:len(confidence_scores)]],
            mode='markers',
            marker=dict(
                color=[self.color_map.get(s, '#808080') for s in sentiment_labels[:len(confidence_scores)]],
                size=8,
                opacity=0.7
            ),
            name="Sentiment vs Confidence"
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Batch Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_word_sentiment_chart(self, word_analysis):
        """
        Create visualization for word sentiment analysis
        """
        if not word_analysis or 'error' in word_analysis:
            return go.Figure()
        
        # Count words by sentiment
        pos_count = len(word_analysis.get('positive_words', []))
        neg_count = len(word_analysis.get('negative_words', []))
        neu_count = len(word_analysis.get('neutral_words', []))
        
        # Create bar chart
        sentiments = ['Positive', 'Negative', 'Neutral']
        counts = [pos_count, neg_count, neu_count]
        colors = [self.color_map['positive'], self.color_map['negative'], self.color_map['neutral']]
        
        fig = go.Figure(data=[go.Bar(
            x=sentiments,
            y=counts,
            marker=dict(color=colors),
            text=counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Word Count by Sentiment Category",
            xaxis_title="Sentiment",
            yaxis_title="Number of Words",
            height=400
        )
        
        return fig
    
    def create_top_words_chart(self, word_analysis, sentiment_type='positive', top_n=10):
        """
        Create horizontal bar chart for top sentiment words
        """
        if not word_analysis or f'{sentiment_type}_words' not in word_analysis:
            return go.Figure()
        
        words_data = word_analysis[f'{sentiment_type}_words']
        if not words_data:
            return go.Figure()
        
        # Get top N words
        top_words = words_data[:top_n]
        words = [item['word'] for item in top_words]
        scores = [abs(item['compound_score']) for item in top_words]
        
        # Choose color based on sentiment type
        color = self.color_map.get(sentiment_type, '#808080')
        
        fig = go.Figure(data=[go.Bar(
            x=scores,
            y=words,
            orientation='h',
            marker=dict(color=color, opacity=0.7),
            text=[f"{score:.3f}" for score in scores],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f"Top {top_n} {sentiment_type.title()} Words by Strength",
            xaxis_title="Sentiment Strength",
            yaxis_title="Words",
            height=max(300, len(words) * 25),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
