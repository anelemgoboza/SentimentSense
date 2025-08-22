# Comprehensive Sentiment Analysis Dashboard

A no-code/low-code sentiment analysis solution built with Streamlit that provides enterprise-level functionality for text sentiment analysis with multiple input methods, interactive dashboards, and comprehensive export capabilities.

## Features

### 🎯 Multi-Input Text Processing
- **Direct Text Entry**: Type or paste text directly into the application
- **File Upload**: Upload `.txt` files for individual analysis
- **Batch Processing**: Process multiple texts via CSV upload, multiple file upload, or manual entry

### 🧠 Advanced Sentiment Analysis
- **TextBlob Integration**: Polarity-based sentiment classification with subjectivity scoring
- **VADER Sentiment**: Specialized for social media text with compound scoring
- **Consensus Algorithm**: Combines multiple analysis engines for improved accuracy
- **Confidence Scoring**: Provides confidence metrics for all classifications

### 📊 Interactive Dashboard
- Real-time sentiment analysis results
- Historical trend visualization
- Sentiment distribution charts
- Confidence score analysis
- Comparative analysis between different engines

### 📈 Comprehensive Visualizations
- Pie charts for sentiment distribution
- Histograms for confidence score analysis
- Timeline plots for historical trends
- Confusion matrices for performance analysis
- Batch analysis summary dashboards

### 💾 Export Capabilities
- **JSON Format**: Complete analysis data with metadata
- **CSV Format**: Tabular format for spreadsheet analysis
- **Performance Reports**: Accuracy metrics and statistical summaries

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Internet connection for package installation

### Installation
1. Clone or download the application files
2. Install required dependencies (handled automatically)
3. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```

### Usage

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

## Technical Architecture

### Core Components
- **Streamlit Framework**: Web application framework
- **Sentiment Analysis**: TextBlob and VADER libraries
- **Data Processing**: Pandas for data manipulation
- **Visualization**: Plotly for interactive charts
- **Export System**: Multiple format support

### File Structure
