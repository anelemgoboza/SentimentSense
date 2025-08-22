# Overview

This is a comprehensive sentiment analysis dashboard built with Streamlit that provides a no-code/low-code solution for enterprise-level text sentiment analysis. The application features multiple input methods (direct text entry, file uploads, batch processing), advanced sentiment analysis using multiple engines (TextBlob and VADER), and interactive visualizations for analyzing sentiment trends and distributions.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Web-based interface providing reactive UI components with automatic state management
- **Multi-page Application**: Structured with caching decorators for performance optimization
- **Session State Management**: Maintains analysis history and current results across user interactions

## Backend Architecture
- **Modular Component Design**: Separated into distinct classes for sentiment analysis, data processing, and visualization
- **Multiple Analysis Engines**: 
  - TextBlob for polarity-based sentiment classification with subjectivity scoring
  - VADER for social media optimized sentiment analysis with compound scoring
  - Consensus algorithm combining multiple engines for improved accuracy
- **Data Processing Pipeline**: Text preprocessing, file handling (CSV, TXT, JSON), and batch processing capabilities

## Data Storage Solutions
- **Session-based Storage**: Uses Streamlit's session state for temporary data persistence
- **File-based Processing**: Handles multiple file formats without persistent database storage
- **Export Capabilities**: JSON and CSV export functionality for analysis results

## Visualization Architecture
- **Plotly Integration**: Interactive charts and graphs including pie charts, histograms, and timeline plots
- **Real-time Updates**: Dynamic visualization updates based on analysis results
- **Customizable Color Schemes**: Consistent color mapping for sentiment categories

## Error Handling and Validation
- **Robust Text Preprocessing**: Handles various text formats and encoding issues
- **Exception Management**: Comprehensive error handling for file processing and analysis failures
- **Input Validation**: Ensures data quality before processing

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and CSV processing
- **Plotly**: Interactive visualization and charting
- **TextBlob**: Natural language processing and sentiment analysis
- **vaderSentiment**: Social media optimized sentiment analysis

## Data Processing Dependencies
- **NumPy**: Numerical computations for analysis algorithms
- **JSON**: Data serialization and export functionality
- **IO utilities**: File handling and stream processing
- **Datetime**: Timestamp management for analysis history

## Visualization Dependencies
- **Plotly Express**: Simplified plotting interface
- **Plotly Graph Objects**: Advanced chart customization
- **Plotly Subplots**: Multi-panel visualization layouts

## Optional Integrations
- **ReportLab/WeasyPrint**: PDF export functionality (placeholder implementation)
- **Additional ML APIs**: Framework supports integration of external sentiment analysis services