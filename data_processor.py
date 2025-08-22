import pandas as pd
import json
import csv
import io
from datetime import datetime
import re

class DataProcessor:
    """
    Data processing utilities for sentiment analysis
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'txt', 'json']
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text for analysis
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Remove control characters but keep newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def read_csv_file(self, file_content, text_column='text'):
        """
        Read and process CSV file for batch analysis
        """
        try:
            # Convert bytes to string if necessary
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            # Read CSV
            df = pd.read_csv(io.StringIO(file_content))
            
            # Check if text column exists
            if text_column not in df.columns:
                available_columns = list(df.columns)
                return None, f"Column '{text_column}' not found. Available columns: {available_columns}"
            
            # Extract and preprocess texts
            texts = []
            for idx, text in enumerate(df[text_column]):
                if pd.notna(text):
                    processed_text = self.preprocess_text(text)
                    if processed_text:
                        texts.append({
                            'id': idx,
                            'text': processed_text,
                            'original_text': str(text)
                        })
            
            return texts, f"Successfully processed {len(texts)} texts from CSV"
            
        except Exception as e:
            return None, f"Error processing CSV file: {str(e)}"
    
    def read_text_files(self, files):
        """
        Process multiple text files
        """
        texts = []
        errors = []
        
        for i, file in enumerate(files):
            try:
                # Read file content
                if hasattr(file, 'read'):
                    content = file.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                else:
                    content = str(file)
                
                processed_text = self.preprocess_text(content)
                if processed_text:
                    texts.append({
                        'id': i,
                        'text': processed_text,
                        'filename': getattr(file, 'name', f'file_{i}'),
                        'original_text': content
                    })
                
            except Exception as e:
                errors.append(f"Error processing file {getattr(file, 'name', f'file_{i}')}: {str(e)}")
        
        success_msg = f"Successfully processed {len(texts)} text files"
        if errors:
            success_msg += f". Errors: {'; '.join(errors)}"
        
        return texts, success_msg
    
    def parse_manual_texts(self, text_input):
        """
        Parse manually entered texts (one per line)
        """
        if not text_input:
            return [], "No text provided"
        
        lines = text_input.split('\n')
        texts = []
        
        for i, line in enumerate(lines):
            processed_text = self.preprocess_text(line)
            if processed_text:
                texts.append({
                    'id': i,
                    'text': processed_text,
                    'original_text': line.strip()
                })
        
        return texts, f"Successfully processed {len(texts)} manually entered texts"
    
    def export_to_csv(self, results, include_original=True):
        """
        Export analysis results to CSV format
        """
        data = []
        
        for i, result in enumerate(results):
            row = {
                'id': i + 1,
                'text_preview': result['text'][:100] + "..." if len(result['text']) > 100 else result['text'],
                'text_length': len(result['text']),
                'timestamp': datetime.now().isoformat()
            }
            
            if include_original:
                row['original_text'] = result.get('original_text', result['text'])
            
            # Add analysis results
            analysis_results = result.get('results', {})
            
            if 'textblob' in analysis_results:
                tb = analysis_results['textblob']
                row['textblob_sentiment'] = tb.get('sentiment', 'N/A')
                row['textblob_polarity'] = tb.get('polarity', 'N/A')
                row['textblob_subjectivity'] = tb.get('subjectivity', 'N/A')
                row['textblob_confidence'] = tb.get('confidence', 'N/A')
            
            if 'vader' in analysis_results:
                vader = analysis_results['vader']
                row['vader_sentiment'] = vader.get('sentiment', 'N/A')
                row['vader_compound'] = vader.get('compound', 'N/A')
                if 'scores' in vader:
                    scores = vader['scores']
                    row['vader_positive'] = scores.get('pos', 'N/A')
                    row['vader_neutral'] = scores.get('neu', 'N/A')
                    row['vader_negative'] = scores.get('neg', 'N/A')
            
            if 'consensus' in analysis_results:
                consensus = analysis_results['consensus']
                row['consensus_sentiment'] = consensus.get('sentiment', 'N/A')
                row['consensus_confidence'] = consensus.get('confidence', 'N/A')
                row['consensus_agreement'] = consensus.get('agreement_ratio', 'N/A')
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def export_to_json(self, results, include_metadata=True):
        """
        Export analysis results to JSON format
        """
        export_data = {
            'results': results,
            'metadata': {
                'total_texts': len(results),
                'export_timestamp': datetime.now().isoformat(),
                'format_version': '1.0'
            } if include_metadata else {}
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def calculate_statistics(self, results):
        """
        Calculate summary statistics from analysis results
        """
        if not results:
            return {}
        
        stats = {
            'total_texts': len(results),
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'confidence_scores': [],
            'text_lengths': [],
            'processing_errors': 0
        }
        
        for result in results:
            # Text length
            stats['text_lengths'].append(len(result.get('text', '')))
            
            # Check for processing errors
            analysis_results = result.get('results', {})
            if 'error' in analysis_results or not analysis_results:
                stats['processing_errors'] += 1
                continue
            
            # Sentiment distribution and confidence
            if 'consensus' in analysis_results:
                consensus = analysis_results['consensus']
                sentiment = consensus.get('sentiment', 'neutral')
                confidence = consensus.get('confidence', 0)
                
                stats['sentiment_distribution'][sentiment] += 1
                stats['confidence_scores'].append(confidence)
        
        # Calculate averages
        if stats['confidence_scores']:
            stats['average_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            stats['min_confidence'] = min(stats['confidence_scores'])
            stats['max_confidence'] = max(stats['confidence_scores'])
        
        if stats['text_lengths']:
            stats['average_text_length'] = sum(stats['text_lengths']) / len(stats['text_lengths'])
            stats['min_text_length'] = min(stats['text_lengths'])
            stats['max_text_length'] = max(stats['text_lengths'])
        
        # Calculate percentages
        total_analyzed = stats['total_texts'] - stats['processing_errors']
        if total_analyzed > 0:
            for sentiment in stats['sentiment_distribution']:
                count = stats['sentiment_distribution'][sentiment]
                stats['sentiment_distribution'][sentiment] = {
                    'count': count,
                    'percentage': (count / total_analyzed) * 100
                }
        
        return stats
    
    def validate_batch_input(self, texts, max_texts=1000, max_length=10000):
        """
        Validate batch input for processing
        """
        if not texts:
            return False, "No texts provided for analysis"
        
        if len(texts) > max_texts:
            return False, f"Too many texts (max {max_texts})"
        
        for i, text_item in enumerate(texts):
            text = text_item.get('text', '') if isinstance(text_item, dict) else str(text_item)
            
            if len(text) > max_length:
                return False, f"Text {i+1} is too long (max {max_length} characters)"
            
            if len(text.strip()) < 3:
                return False, f"Text {i+1} is too short for meaningful analysis"
        
        return True, "Batch input is valid"
