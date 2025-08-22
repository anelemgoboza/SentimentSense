import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sentiment_analyzer import SentimentAnalyzer
from data_processor import DataProcessor
from visualization import SentimentVisualizer
from utils import export_to_pdf, create_confusion_matrix

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def initialize_processor():
    return DataProcessor()

@st.cache_resource
def initialize_visualizer():
    return SentimentVisualizer()

analyzer = initialize_analyzer()
processor = DataProcessor()
visualizer = SentimentVisualizer()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# Main app
def main():
    st.title("🎯 Comprehensive Sentiment Analysis Dashboard")
    st.markdown("**A no-code/low-code solution for enterprise-level sentiment analysis**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Single Text Analysis", "Word Analysis", "Batch Processing", "Analytics Dashboard", "Performance Analysis", "Documentation"]
    )
    
    if page == "Single Text Analysis":
        single_text_analysis()
    elif page == "Word Analysis":
        word_analysis_page()
    elif page == "Batch Processing":
        batch_processing()
    elif page == "Analytics Dashboard":
        analytics_dashboard()
    elif page == "Performance Analysis":
        performance_analysis()
    elif page == "Documentation":
        documentation()

def single_text_analysis():
    st.header("📝 Single Text Analysis")
    
    # Text input methods
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Entry", "Text File Upload"]
    )
    
    text_to_analyze = ""
    
    if input_method == "Direct Text Entry":
        text_to_analyze = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt'],
            help="Upload a .txt file containing the text to analyze"
        )
        if uploaded_file:
            text_to_analyze = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", value=text_to_analyze, height=150, disabled=True)
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        use_textblob = st.checkbox("TextBlob Analysis", value=True)
    with col2:
        use_vader = st.checkbox("VADER Analysis", value=True)
    
    if st.button("Analyze Sentiment", type="primary"):
        if text_to_analyze.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    results = analyzer.analyze_text(text_to_analyze, use_textblob, use_vader)
                    
                    # Store in session state
                    st.session_state.current_results = results
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'text': text_to_analyze[:100] + "..." if len(text_to_analyze) > 100 else text_to_analyze,
                        'results': results
                    })
                    
                    # Display results
                    display_single_analysis_results(results, text_to_analyze)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

def display_word_analysis(word_analysis):
    """Display word-level sentiment analysis results"""
    st.subheader("🔍 Word-Level Sentiment Analysis")
    
    if 'error' in word_analysis:
        st.error(word_analysis['error'])
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Positive Words", len(word_analysis.get('positive_words', [])))
    with col2:
        st.metric("Negative Words", len(word_analysis.get('negative_words', [])))
    with col3:
        st.metric("Neutral Words", len(word_analysis.get('neutral_words', [])))
    
    # Display word categories in tabs
    tab1, tab2, tab3 = st.tabs(["🟢 Positive Words", "🔴 Negative Words", "⚫ Neutral Words"])
    
    with tab1:
        positive_words = word_analysis.get('positive_words', [])
        if positive_words:
            st.write("**Top Positive Words (by sentiment strength):**")
            pos_df = pd.DataFrame(positive_words[:15])  # Show top 15
            pos_df = pos_df[['word', 'compound_score', 'pos_score']].round(3)
            pos_df.columns = ['Word', 'Overall Score', 'Positive Score']
            st.dataframe(pos_df, use_container_width=True)
            
            # Word cloud style display
            st.write("**All Positive Words:**")
            pos_word_list = [word['word'] for word in positive_words]
            st.write(" • ".join(pos_word_list))
        else:
            st.info("No positive words detected in this text.")
    
    with tab2:
        negative_words = word_analysis.get('negative_words', [])
        if negative_words:
            st.write("**Top Negative Words (by sentiment strength):**")
            neg_df = pd.DataFrame(negative_words[:15])  # Show top 15
            neg_df = neg_df[['word', 'compound_score', 'neg_score']].round(3)
            neg_df.columns = ['Word', 'Overall Score', 'Negative Score']
            st.dataframe(neg_df, use_container_width=True)
            
            # Word cloud style display
            st.write("**All Negative Words:**")
            neg_word_list = [word['word'] for word in negative_words]
            st.write(" • ".join(neg_word_list))
        else:
            st.info("No negative words detected in this text.")
    
    with tab3:
        neutral_words = word_analysis.get('neutral_words', [])
        if neutral_words:
            st.write(f"**Neutral Words ({len(neutral_words)} total):**")
            # Show sample of neutral words
            neutral_sample = neutral_words[:30]  # Show first 30
            neutral_word_list = [word['word'] for word in neutral_sample]
            st.write(" • ".join(neutral_word_list))
            if len(neutral_words) > 30:
                st.caption(f"... and {len(neutral_words) - 30} more neutral words")
        else:
            st.info("No neutral words detected in this text.")

def display_phrase_analysis(phrase_analysis):
    """Display phrase-level sentiment analysis results"""
    st.subheader("📝 Phrase-Level Sentiment Analysis")
    
    if not phrase_analysis:
        st.info("No significant sentiment-bearing phrases detected.")
        return
    
    st.write("**Sentences/phrases with strong sentiment:**")
    
    for i, phrase in enumerate(phrase_analysis[:10]):  # Show top 10 phrases
        sentiment_color = "🟢" if phrase['sentiment'] == 'positive' else "🔴"
        
        with st.expander(f"{sentiment_color} {phrase['sentiment'].title()} (Strength: {phrase['strength']:.3f})"):
            st.write(f"**Phrase:** {phrase['phrase']}")
            st.write(f"**Polarity Score:** {phrase['polarity']:.3f}")
            st.write(f"**Sentiment Strength:** {phrase['strength']:.3f}")
            
            # Progress bar to show strength
            st.progress(min(phrase['strength'], 1.0))

def word_analysis_page():
    """Dedicated page for word-level sentiment analysis"""
    st.header("🔍 Word-Level Sentiment Analysis")
    st.markdown("**Discover which specific words contribute to positive, negative, and neutral sentiments in your text.**")
    
    # Text input
    st.subheader("Enter Text for Word Analysis")
    
    # Pre-fill with example text if selected
    default_text = ""
    if 'example_text' in st.session_state:
        default_text = st.session_state.example_text
        # Clear after using
        del st.session_state.example_text
    
    text_to_analyze = st.text_area(
        "Enter or paste your text:",
        height=200,
        value=default_text,
        placeholder="Type your text here to see which words are positive, negative, or neutral..."
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        include_phrases = st.checkbox("Include phrase analysis", value=True)
    with col2:
        min_word_length = st.slider("Minimum word length", 1, 10, 3)
    
    if st.button("Analyze Words", type="primary"):
        if text_to_analyze.strip():
            with st.spinner("Analyzing words and their sentiment..."):
                try:
                    # Get word analysis directly
                    word_analysis = analyzer.analyze_word_sentiment(text_to_analyze)
                    
                    # Filter words by minimum length
                    for category in ['positive_words', 'negative_words', 'neutral_words']:
                        if category in word_analysis:
                            word_analysis[category] = [
                                word for word in word_analysis[category] 
                                if len(word['word']) >= min_word_length
                            ]
                    
                    # Display results
                    display_word_analysis(word_analysis)
                    
                    # Add visualizations
                    st.subheader("📊 Word Sentiment Visualizations")
                    
                    # Overall word distribution chart
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_distribution = visualizer.create_word_sentiment_chart(word_analysis)
                        st.plotly_chart(fig_distribution, use_container_width=True)
                    
                    with col2:
                        # Top positive words chart
                        if word_analysis.get('positive_words'):
                            fig_positive = visualizer.create_top_words_chart(word_analysis, 'positive', 8)
                            st.plotly_chart(fig_positive, use_container_width=True)
                    
                    # Top negative words chart
                    if word_analysis.get('negative_words'):
                        fig_negative = visualizer.create_top_words_chart(word_analysis, 'negative', 8)
                        st.plotly_chart(fig_negative, use_container_width=True)
                    
                    # Show phrase analysis if requested
                    if include_phrases:
                        phrase_analysis = analyzer.get_sentiment_phrases(text_to_analyze)
                        display_phrase_analysis(phrase_analysis)
                    
                    # Text statistics
                    st.subheader("📊 Text Statistics")
                    words = text_to_analyze.split()
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Words", len(words))
                    with col2:
                        st.metric("Characters", len(text_to_analyze))
                    with col3:
                        st.metric("Sentences", len([s for s in text_to_analyze.split('.') if s.strip()]))
                    with col4:
                        unique_words = len(set(word.lower().strip('.,!?;:"()[]{}') for word in words))
                        st.metric("Unique Words", unique_words)
                    
                except Exception as e:
                    st.error(f"Error during word analysis: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Example texts
    st.subheader("💡 Try These Examples")
    examples = [
        {
            "title": "Product Review",
            "text": "This product is absolutely amazing! The quality is excellent and the customer service was fantastic. However, the price is quite expensive and delivery was slow."
        },
        {
            "title": "Restaurant Review", 
            "text": "The food was delicious and the atmosphere was wonderful. The staff was friendly and helpful. Unfortunately, the service was terrible and we had to wait a very long time."
        },
        {
            "title": "Movie Review",
            "text": "This movie was boring and disappointing. The plot was confusing and the acting was poor. I hate wasting time on bad films like this one."
        }
    ]
    
    for example in examples:
        with st.expander(f"📝 Example: {example['title']}"):
            st.write(example['text'])
            if st.button(f"Analyze this text", key=f"example_{example['title']}"):
                # Store example text in session state for use
                st.session_state.example_text = example['text']
                st.rerun()

def display_single_analysis_results(results, original_text):
    st.subheader("Analysis Results")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    if results.get('textblob'):
        tb_sentiment = results['textblob']['sentiment']
        with col1:
            st.metric(
                "TextBlob Sentiment",
                tb_sentiment.capitalize(),
                f"Confidence: {results['textblob']['confidence']:.2f}"
            )
    
    if results.get('vader'):
        vader_sentiment = results['vader']['sentiment']
        with col2:
            st.metric(
                "VADER Sentiment",
                vader_sentiment.capitalize(),
                f"Score: {results['vader']['compound']:.2f}"
            )
    
    if results.get('consensus'):
        with col3:
            st.metric(
                "Consensus",
                results['consensus']['sentiment'].capitalize(),
                f"Confidence: {results['consensus']['confidence']:.2f}"
            )
    
    # Detailed scores
    if results.get('vader'):
        st.subheader("Detailed VADER Scores")
        vader_scores = results['vader']['scores']
        
        score_cols = st.columns(4)
        with score_cols[0]:
            st.metric("Positive", f"{vader_scores['pos']:.3f}")
        with score_cols[1]:
            st.metric("Neutral", f"{vader_scores['neu']:.3f}")
        with score_cols[2]:
            st.metric("Negative", f"{vader_scores['neg']:.3f}")
        with score_cols[3]:
            st.metric("Compound", f"{vader_scores['compound']:.3f}")
    
    # Visualization
    st.subheader("Sentiment Visualization")
    if results.get('vader'):
        fig = visualizer.create_sentiment_pie_chart(results['vader']['scores'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Word-level sentiment analysis
    if results.get('word_analysis'):
        display_word_analysis(results['word_analysis'])
    
    # Phrase-level sentiment analysis
    if results.get('phrase_analysis'):
        display_phrase_analysis(results['phrase_analysis'])
    
    # Export options
    st.subheader("Export Results")
    export_cols = st.columns(3)
    
    with export_cols[0]:
        if st.button("Export as JSON"):
            json_data = json.dumps({
                'text': original_text,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    with export_cols[1]:
        if st.button("Export as CSV"):
            df = pd.DataFrame([{
                'text': original_text[:50] + "..." if len(original_text) > 50 else original_text,
                'textblob_sentiment': results.get('textblob', {}).get('sentiment', 'N/A'),
                'textblob_confidence': results.get('textblob', {}).get('confidence', 'N/A'),
                'vader_sentiment': results.get('vader', {}).get('sentiment', 'N/A'),
                'vader_compound': results.get('vader', {}).get('compound', 'N/A'),
                'consensus_sentiment': results.get('consensus', {}).get('sentiment', 'N/A'),
                'consensus_confidence': results.get('consensus', {}).get('confidence', 'N/A'),
                'timestamp': datetime.now().isoformat()
            }])
            csv_data = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

def batch_processing():
    st.header("📂 Batch Processing")
    st.markdown("Upload multiple texts for simultaneous sentiment analysis.")
    
    # File upload options
    upload_type = st.radio(
        "Choose upload type:",
        ["CSV File", "Text Files", "Manual Entry"]
    )
    
    texts_to_analyze = []
    
    if upload_type == "CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should have a 'text' column containing the texts to analyze"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts_to_analyze = df['text'].dropna().tolist()
                    st.success(f"Loaded {len(texts_to_analyze)} texts from CSV")
                    st.dataframe(df.head())
                else:
                    st.error("CSV file must contain a 'text' column")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    elif upload_type == "Text Files":
        uploaded_files = st.file_uploader(
            "Upload text files",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload multiple .txt files"
        )
        if uploaded_files:
            for file in uploaded_files:
                try:
                    content = str(file.read(), "utf-8")
                    texts_to_analyze.append(content)
                except Exception as e:
                    st.error(f"Error reading file {file.name}: {str(e)}")
            if texts_to_analyze:
                st.success(f"Loaded {len(texts_to_analyze)} text files")
    
    else:  # Manual Entry
        st.markdown("Enter multiple texts (one per line):")
        manual_texts = st.text_area(
            "Texts to analyze:",
            height=200,
            placeholder="Enter each text on a new line..."
        )
        if manual_texts:
            texts_to_analyze = [text.strip() for text in manual_texts.split('\n') if text.strip()]
            st.info(f"Found {len(texts_to_analyze)} texts to analyze")
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    with col1:
        use_textblob = st.checkbox("TextBlob Analysis", value=True, key="batch_textblob")
    with col2:
        use_vader = st.checkbox("VADER Analysis", value=True, key="batch_vader")
    with col3:
        show_progress = st.checkbox("Show Progress", value=True)
    
    if st.button("Start Batch Analysis", type="primary"):
        if texts_to_analyze:
            batch_results = []
            
            progress_bar = None
            status_text = None
            if show_progress:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            for i, text in enumerate(texts_to_analyze):
                if show_progress and progress_bar is not None and status_text is not None:
                    progress = (i + 1) / len(texts_to_analyze)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing text {i + 1} of {len(texts_to_analyze)}")
                
                try:
                    result = analyzer.analyze_text(text, use_textblob, use_vader)
                    batch_results.append({
                        'text': text,
                        'results': result
                    })
                except Exception as e:
                    st.warning(f"Error analyzing text {i + 1}: {str(e)}")
            
            if show_progress and progress_bar is not None and status_text is not None:
                progress_bar.empty()
                status_text.empty()
            
            # Store results
            st.session_state.batch_results = batch_results
            
            # Display batch results
            display_batch_results(batch_results)
        else:
            st.warning("Please provide texts to analyze.")

def display_batch_results(batch_results):
    st.subheader("Batch Analysis Results")
    
    if not batch_results:
        st.warning("No results to display.")
        return
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    # Calculate sentiment distribution
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    confidence_scores = []
    
    for result in batch_results:
        if result['results'].get('consensus'):
            sentiment = result['results']['consensus']['sentiment']
            confidence = result['results']['consensus']['confidence']
            sentiment_counts[sentiment] += 1
            confidence_scores.append(confidence)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Texts", len(batch_results))
    with col2:
        st.metric("Positive", sentiment_counts['positive'])
    with col3:
        st.metric("Negative", sentiment_counts['negative'])
    with col4:
        st.metric("Neutral", sentiment_counts['neutral'])
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        fig_pie = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence score distribution
        if confidence_scores:
            fig_hist = px.histogram(
                x=confidence_scores,
                nbins=20,
                title="Confidence Score Distribution"
            )
            fig_hist.update_xaxes(title="Confidence Score")
            fig_hist.update_yaxes(title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    # Create DataFrame for display
    results_data = []
    for i, result in enumerate(batch_results):
        row = {
            'ID': i + 1,
            'Text Preview': result['text'][:100] + "..." if len(result['text']) > 100 else result['text'],
            'Length': len(result['text'])
        }
        
        if result['results'].get('textblob'):
            row['TextBlob Sentiment'] = result['results']['textblob']['sentiment']
            row['TextBlob Confidence'] = f"{result['results']['textblob']['confidence']:.3f}"
        
        if result['results'].get('vader'):
            row['VADER Sentiment'] = result['results']['vader']['sentiment']
            row['VADER Compound'] = f"{result['results']['vader']['compound']:.3f}"
        
        if result['results'].get('consensus'):
            row['Consensus'] = result['results']['consensus']['sentiment']
            row['Consensus Confidence'] = f"{result['results']['consensus']['confidence']:.3f}"
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # Export options
    st.subheader("Export Batch Results")
    export_cols = st.columns(3)
    
    with export_cols[0]:
        if st.button("Export as JSON", key="batch_json"):
            json_data = json.dumps({
                'batch_results': batch_results,
                'summary': {
                    'total_texts': len(batch_results),
                    'sentiment_distribution': sentiment_counts,
                    'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                },
                'timestamp': datetime.now().isoformat()
            }, indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                f"batch_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    with export_cols[1]:
        if st.button("Export as CSV", key="batch_csv"):
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                f"batch_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

def analytics_dashboard():
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Perform some sentiment analysis first.")
        return
    
    # Historical trends
    st.subheader("Historical Analysis Trends")
    
    # Prepare data for visualization
    history_data = []
    for entry in st.session_state.analysis_history:
        timestamp = entry['timestamp']
        results = entry['results']
        
        if results.get('consensus'):
            history_data.append({
                'timestamp': timestamp,
                'sentiment': results['consensus']['sentiment'],
                'confidence': results['consensus']['confidence'],
                'text_preview': entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text']
            })
    
    if history_data:
        df_history = pd.DataFrame(history_data)
        
        # Time series of sentiment
        fig_timeline = px.scatter(
            df_history,
            x='timestamp',
            y='confidence',
            color='sentiment',
            title='Sentiment Analysis Timeline',
            hover_data=['text_preview']
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Sentiment distribution over time
        sentiment_by_hour = df_history.groupby([
            df_history['timestamp'].dt.hour,
            'sentiment'
        ]).size().reset_index()
        sentiment_by_hour.columns = [sentiment_by_hour.columns[0], sentiment_by_hour.columns[1], 'count']
        
        fig_hourly = px.bar(
            sentiment_by_hour,
            x='timestamp',
            y='count',
            color='sentiment',
            title='Sentiment Distribution by Hour'
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Display recent analyses
        st.subheader("Recent Analyses")
        recent_df = df_history.tail(10)[['timestamp', 'sentiment', 'confidence', 'text_preview']]
        st.dataframe(recent_df, use_container_width=True)

def performance_analysis():
    st.header("⚡ Performance Analysis")
    
    st.markdown("""
    ### Platform Analysis: Streamlit for Sentiment Analysis
    
    **Advantages of this No-Code/Low-Code Approach:**
    
    ✅ **Rapid Prototyping**: Streamlit enables quick development and deployment of ML applications without complex frontend development.
    
    ✅ **Integration Flexibility**: Easy integration with multiple sentiment analysis libraries (TextBlob, VADER) and potential for API connections.
    
    ✅ **Interactive Dashboards**: Built-in support for interactive visualizations using Plotly, making data exploration intuitive.
    
    ✅ **Real-time Processing**: Immediate feedback and results display enhances user experience.
    
    ✅ **Export Capabilities**: Multiple export formats (CSV, JSON, PDF) for further analysis and reporting.
    
    **Limitations and Trade-offs:**
    
    ⚠️ **Customization Constraints**: Limited UI customization compared to full-stack development approaches.
    
    ⚠️ **Scalability Considerations**: Session-based state management may not be optimal for high-concurrency scenarios.
    
    ⚠️ **Deployment Dependencies**: Requires Python environment and specific dependencies, unlike pure no-code platforms.
    
    ⚠️ **Advanced ML Features**: Limited built-in ML model training capabilities compared to specialized ML platforms.
    """)
    
    # Performance metrics
    if st.session_state.analysis_history:
        st.subheader("Usage Statistics")
        
        total_analyses = len(st.session_state.analysis_history)
        
        # Calculate accuracy if manual labels were provided
        accuracy_data = []
        for entry in st.session_state.analysis_history:
            if entry['results'].get('consensus'):
                accuracy_data.append(entry['results']['consensus']['confidence'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            avg_confidence = sum(accuracy_data) / len(accuracy_data) if accuracy_data else 0
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        with col3:
            high_confidence = sum(1 for conf in accuracy_data if conf > 0.7)
            st.metric("High Confidence Results", f"{high_confidence}/{len(accuracy_data)}")
        
        # Confusion matrix simulation (would need manual labels for real implementation)
        st.subheader("Model Performance Simulation")
        st.markdown("*Note: This would require manual sentiment labels for true accuracy calculation*")
        
        # Create sample confusion matrix
        confusion_fig = create_confusion_matrix()
        st.plotly_chart(confusion_fig, use_container_width=True)

def documentation():
    st.header("📚 Documentation")
    
    st.markdown("""
    ## Comprehensive Sentiment Analysis Solution
    
    ### Overview
    This application demonstrates a no-code/low-code approach to sentiment analysis using Streamlit as the primary platform. It provides enterprise-level functionality for text sentiment analysis with minimal traditional programming.
    
    ### Features
    
    #### 1. Multiple Input Methods
    - **Direct Text Entry**: Type or paste text directly into the application
    - **File Upload**: Upload text files (.txt) for analysis
    - **Batch Processing**: Analyze multiple texts simultaneously via CSV upload or manual entry
    
    #### 2. Multi-Engine Analysis
    - **TextBlob**: Provides polarity-based sentiment classification
    - **VADER**: Specialized for social media text with compound scoring
    - **Consensus Algorithm**: Combines results from multiple engines for improved accuracy
    
    #### 3. Interactive Dashboard
    - Real-time sentiment analysis results
    - Historical trend visualization
    - Confidence score distributions
    - Sentiment distribution charts
    
    #### 4. Export Capabilities
    - **JSON**: Complete analysis data with metadata
    - **CSV**: Tabular format for spreadsheet analysis
    - **PDF**: Formatted reports (future enhancement)
    
    ### User Guide
    
    #### Single Text Analysis
    1. Navigate to "Single Text Analysis" page
    2. Choose input method (Direct Entry or File Upload)
    3. Enter or upload your text
    4. Select analysis engines (TextBlob and/or VADER)
    5. Click "Analyze Sentiment"
    6. Review results and export if needed
    
    #### Batch Processing
    1. Go to "Batch Processing" page
    2. Choose upload type (CSV, Text Files, or Manual Entry)
    3. Provide your texts using the selected method
    4. Configure analysis options
    5. Start batch analysis
    6. Review summary statistics and detailed results
    7. Export batch results
    
    #### Analytics Dashboard
    1. Access "Analytics Dashboard" to view historical trends
    2. Examine sentiment patterns over time
    3. Review recent analysis history
    
    ### Technical Implementation
    
    #### Architecture
    - **Frontend**: Streamlit web framework
    - **Sentiment Analysis**: TextBlob and VADER libraries
    - **Visualization**: Plotly for interactive charts
    - **Data Processing**: Pandas for data manipulation
    
    #### Error Handling
    - Input validation for text content
    - File format verification
    - Graceful handling of analysis failures
    - User-friendly error messages
    
    ### Platform Justification
    
    **Why Streamlit for No-Code/Low-Code Sentiment Analysis:**
    
    1. **Rapid Development**: Streamlit allows creation of sophisticated ML applications with minimal code
    2. **Python Ecosystem**: Direct access to powerful ML libraries without API limitations
    3. **Interactive UI**: Built-in widgets and real-time updates enhance user experience
    4. **Deployment Simplicity**: Single-command deployment with automatic port configuration
    5. **Extensibility**: Easy integration with external APIs and services
    
    ### Performance Considerations
    
    - **Session State Management**: Maintains analysis history during user session
    - **Caching**: Strategic use of Streamlit's caching for performance optimization
    - **Memory Efficiency**: Optimized data structures for batch processing
    
    ### Future Enhancements
    
    - Integration with cloud-based sentiment analysis APIs
    - Real-time social media monitoring capabilities
    - User authentication and role-based access control
    - Mobile-responsive interface improvements
    - Advanced ML model training capabilities
    
    ### Limitations
    
    - Session-based storage (data doesn't persist between sessions)
    - Limited to pre-trained models (TextBlob, VADER)
    - UI customization constraints compared to full-stack development
    - Requires Python environment for deployment
    
    ### Support and Maintenance
    
    This application demonstrates the power of no-code/low-code platforms for rapid ML solution development while maintaining professional-grade functionality and user experience.
    """)

if __name__ == "__main__":
    main()
