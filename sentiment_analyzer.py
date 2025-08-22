import os
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis class supporting multiple engines
    """
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        # Common positive and negative words for basic classification
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'awesome',
            'brilliant', 'outstanding', 'perfect', 'love', 'like', 'enjoy', 'happy',
            'pleased', 'satisfied', 'delighted', 'thrilled', 'excited', 'impressed',
            'beautiful', 'nice', 'pleasant', 'positive', 'recommend', 'best'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'dislike',
            'disappointed', 'frustrated', 'angry', 'annoyed', 'upset', 'sad',
            'disgusting', 'pathetic', 'useless', 'boring', 'slow', 'expensive',
            'poor', 'weak', 'broken', 'failed', 'wrong', 'problem', 'issue'
        }
        
    def analyze_with_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        Returns sentiment classification and confidence score
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to sentiment class
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Calculate confidence based on absolute polarity
            confidence = abs(polarity)
            if confidence == 0:
                confidence = 0.5  # Neutral confidence for zero polarity
            
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'confidence': confidence
            }
        except Exception as e:
            raise Exception(f"TextBlob analysis failed: {str(e)}")
    
    def analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER
        Returns sentiment classification and detailed scores
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Convert compound score to sentiment class
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'compound': compound,
                'scores': {
                    'pos': scores['pos'],
                    'neu': scores['neu'],
                    'neg': scores['neg'],
                    'compound': compound
                }
            }
        except Exception as e:
            raise Exception(f"VADER analysis failed: {str(e)}")
    
    def analyze_with_huggingface(self, text, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Analyze sentiment using Hugging Face API
        Requires HF_API_KEY environment variable
        """
        api_key = os.getenv("HF_API_KEY", "")
        if not api_key:
            raise Exception("Hugging Face API key not found in environment variables")
        
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": text},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    predictions = result[0]
                    
                    # Find highest scoring prediction
                    best_prediction = max(predictions, key=lambda x: x['score'])
                    
                    # Map labels to standard format
                    label_mapping = {
                        'LABEL_0': 'negative',
                        'LABEL_1': 'neutral', 
                        'LABEL_2': 'positive',
                        'NEGATIVE': 'negative',
                        'NEUTRAL': 'neutral',
                        'POSITIVE': 'positive'
                    }
                    
                    sentiment = label_mapping.get(best_prediction['label'], best_prediction['label'].lower())
                    confidence = best_prediction['score']
                    
                    return {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'all_predictions': predictions
                    }
            else:
                raise Exception(f"API request failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Hugging Face API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Hugging Face analysis failed: {str(e)}")
    
    def create_consensus(self, textblob_result, vader_result, hf_result=None):
        """
        Create consensus sentiment from multiple analyzers
        """
        sentiments = []
        confidences = []
        
        if textblob_result:
            sentiments.append(textblob_result['sentiment'])
            confidences.append(textblob_result['confidence'])
        
        if vader_result:
            sentiments.append(vader_result['sentiment'])
            # Convert VADER compound to confidence-like score
            vader_confidence = abs(vader_result['compound'])
            confidences.append(vader_confidence)
        
        if hf_result:
            sentiments.append(hf_result['sentiment'])
            confidences.append(hf_result['confidence'])
        
        if not sentiments:
            return None
        
        # Count sentiment votes
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for sentiment in sentiments:
            sentiment_counts[sentiment] += 1
        
        # Get consensus sentiment (most votes)
        consensus_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average confidence
        consensus_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Adjust confidence based on agreement
        agreement_ratio = sentiment_counts[consensus_sentiment] / len(sentiments)
        consensus_confidence *= agreement_ratio
        
        return {
            'sentiment': consensus_sentiment,
            'confidence': consensus_confidence,
            'agreement_ratio': agreement_ratio,
            'votes': sentiment_counts
        }
    
    def analyze_text(self, text, use_textblob=True, use_vader=True, use_huggingface=False):
        """
        Comprehensive text analysis using selected methods
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        results = {}
        
        # TextBlob analysis
        if use_textblob:
            try:
                results['textblob'] = self.analyze_with_textblob(text)
            except Exception as e:
                results['textblob_error'] = str(e)
        
        # VADER analysis
        if use_vader:
            try:
                results['vader'] = self.analyze_with_vader(text)
            except Exception as e:
                results['vader_error'] = str(e)
        
        # Hugging Face analysis (optional)
        if use_huggingface:
            try:
                results['huggingface'] = self.analyze_with_huggingface(text)
            except Exception as e:
                results['huggingface_error'] = str(e)
        
        # Create consensus if multiple methods used
        textblob_result = results.get('textblob')
        vader_result = results.get('vader')
        hf_result = results.get('huggingface')
        
        if textblob_result or vader_result or hf_result:
            consensus = self.create_consensus(textblob_result, vader_result, hf_result)
            if consensus:
                results['consensus'] = consensus
        
        # Add word-level analysis
        try:
            results['word_analysis'] = self.analyze_word_sentiment(text)
        except Exception as e:
            results['word_analysis_error'] = str(e)
        
        # Add phrase-level analysis
        try:
            results['phrase_analysis'] = self.get_sentiment_phrases(text)
        except Exception as e:
            results['phrase_analysis_error'] = str(e)
        
        return results
    
    def validate_text(self, text):
        """
        Validate input text for analysis
        """
        if not text:
            return False, "Text cannot be empty"
        
        if len(text.strip()) < 3:
            return False, "Text too short for meaningful analysis"
        
        if len(text) > 10000:
            return False, "Text too long (max 10,000 characters)"
        
        return True, "Text is valid"
    
    def analyze_word_sentiment(self, text):
        """
        Analyze sentiment of individual words in the text
        Returns categorized words and their sentiment scores
        """
        try:
            # Clean and tokenize text
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            word_analysis = {
                'positive_words': [],
                'negative_words': [],
                'neutral_words': [],
                'word_scores': {}
            }
            
            # Analyze each word with VADER
            for word in words:
                if len(word) > 2:  # Skip very short words
                    # Get VADER score for individual word
                    vader_score = self.vader_analyzer.polarity_scores(word)
                    compound = vader_score['compound']
                    
                    word_info = {
                        'word': word,
                        'compound_score': compound,
                        'pos_score': vader_score['pos'],
                        'neg_score': vader_score['neg'],
                        'neu_score': vader_score['neu']
                    }
                    
                    # Classify word based on compound score and known word lists
                    if word in self.positive_words or compound > 0.1:
                        word_analysis['positive_words'].append(word_info)
                        word_info['classification'] = 'positive'
                    elif word in self.negative_words or compound < -0.1:
                        word_analysis['negative_words'].append(word_info)
                        word_info['classification'] = 'negative'
                    else:
                        word_analysis['neutral_words'].append(word_info)
                        word_info['classification'] = 'neutral'
                    
                    word_analysis['word_scores'][word] = word_info
            
            # Sort words by sentiment strength
            word_analysis['positive_words'].sort(key=lambda x: x['compound_score'], reverse=True)
            word_analysis['negative_words'].sort(key=lambda x: x['compound_score'])
            
            # Get most influential words
            word_analysis['top_positive'] = word_analysis['positive_words'][:10]
            word_analysis['top_negative'] = word_analysis['negative_words'][:10]
            
            return word_analysis
            
        except Exception as e:
            return {
                'error': f"Word analysis failed: {str(e)}",
                'positive_words': [],
                'negative_words': [],
                'neutral_words': [],
                'word_scores': {}
            }
    
    def get_sentiment_phrases(self, text):
        """
        Extract sentiment-bearing phrases from text
        """
        try:
            blob = TextBlob(text)
            phrases = []
            
            # Split into sentences
            sentences = blob.sentences
            
            for sentence in sentences:
                sentence_text = str(sentence)
                sentence_polarity = sentence.sentiment.polarity
                
                if abs(sentence_polarity) > 0.1:  # Only significant sentiment
                    phrases.append({
                        'phrase': sentence_text,
                        'polarity': sentence_polarity,
                        'sentiment': 'positive' if sentence_polarity > 0 else 'negative',
                        'strength': abs(sentence_polarity)
                    })
            
            # Sort by sentiment strength
            phrases.sort(key=lambda x: x['strength'], reverse=True)
            
            return phrases
            
        except Exception as e:
            return []
