"""
Swasth AI - Flask Backend
Multilingual disease-awareness chatbot with semantic search
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
import os
import csv
import logging
from typing import List, Dict, Tuple, Optional

# Semantic search libraries
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Falling back to TF-IDF.")

# TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect not available. Using default language.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Flask App Configuration =====
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///swasth_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_SORT_KEYS'] = False

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)

# Database
db = SQLAlchemy(app)

# ===== Database Models =====
class FAQ(db.Model):
    """FAQ knowledge base table"""
    __tablename__ = 'faqs'
    
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(10), default='en')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'language': self.language
        }

class QueryLog(db.Model):
    """Query logging for analytics"""
    __tablename__ = 'query_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_query = db.Column(db.Text, nullable=False)
    detected_language = db.Column(db.String(10))
    matched_faq_id = db.Column(db.Integer, db.ForeignKey('faqs.id'))
    confidence_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_query': self.user_query,
            'detected_language': self.detected_language,
            'matched_faq_id': self.matched_faq_id,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat()
        }

# ===== Semantic Search Engine =====
class SemanticSearchEngine:
    """Handles embedding-based semantic search with TF-IDF fallback"""
    
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.faqs = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.use_embeddings = EMBEDDINGS_AVAILABLE
        
        if self.use_embeddings:
            try:
                logger.info("Loading multilingual sentence transformer model...")
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embeddings model: {e}")
                self.use_embeddings = False
        
        if not self.use_embeddings:
            logger.info("Using TF-IDF fallback for semantic search")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english'
            )
    
    def build_index(self, faqs: List[FAQ]):
        """Build search index from FAQs"""
        self.faqs = faqs
        
        if not faqs:
            logger.warning("No FAQs to index")
            return
        
        # Combine question and answer for better matching
        texts = [f"{faq.question} {faq.answer}" for faq in faqs]
        
        if self.use_embeddings:
            try:
                logger.info(f"Building embeddings for {len(texts)} FAQs...")
                self.embeddings = self.model.encode(texts, show_progress_bar=False)
                logger.info("✅ Embeddings built successfully")
            except Exception as e:
                logger.error(f"Error building embeddings: {e}")
                self.use_embeddings = False
                self._build_tfidf_index(texts)
        else:
            self._build_tfidf_index(texts)
    
    def _build_tfidf_index(self, texts: List[str]):
        """Build TF-IDF index as fallback"""
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("✅ TF-IDF index built successfully")
        except Exception as e:
            logger.error(f"Error building TF-IDF index: {e}")
    
    def search(self, query: str, top_k: int = 1) -> List[Tuple[FAQ, float]]:
        """Search for most relevant FAQs"""
        if not self.faqs:
            return []
        
        if self.use_embeddings and self.embeddings is not None:
            return self._search_embeddings(query, top_k)
        elif self.tfidf_matrix is not None:
            return self._search_tfidf(query, top_k)
        else:
            # Fallback to simple keyword matching
            return self._search_keyword(query, top_k)
    
    def _search_embeddings(self, query: str, top_k: int) -> List[Tuple[FAQ, float]]:
        """Search using embeddings"""
        try:
            query_embedding = self.model.encode([query])[0]
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = [
                (self.faqs[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            return results
        except Exception as e:
            logger.error(f"Embedding search error: {e}")
            return []
    
    def _search_tfidf(self, query: str, top_k: int) -> List[Tuple[FAQ, float]]:
        """Search using TF-IDF"""
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = [
                (self.faqs[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            return results
        except Exception as e:
            logger.error(f"TF-IDF search error: {e}")
            return []
    
    def _search_keyword(self, query: str, top_k: int) -> List[Tuple[FAQ, float]]:
        """Simple keyword matching fallback"""
        query_lower = query.lower()
        scores = []
        
        for faq in self.faqs:
            text = f"{faq.question} {faq.answer}".lower()
            score = sum(1 for word in query_lower.split() if word in text)
            scores.append(score)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = [
            (self.faqs[idx], float(scores[idx]) / max(len(query_lower.split()), 1))
            for idx in top_indices if scores[idx] > 0
        ]
        return results

# Initialize search engine
search_engine = SemanticSearchEngine()

# ===== Helper Functions =====
def detect_language(text: str) -> str:
    """Detect language of input text"""
    if not LANGDETECT_AVAILABLE:
        return 'en'
    
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def load_faqs_from_db():
    """Load FAQs from database and rebuild index"""
    faqs = FAQ.query.all()
    search_engine.build_index(faqs)
    logger.info(f"Loaded {len(faqs)} FAQs from database")

# ===== API Endpoints =====
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'embeddings_enabled': search_engine.use_embeddings,
        'faqs_loaded': len(search_engine.faqs)
    })

@app.route('/query', methods=['POST'])
@limiter.limit("10 per minute")
def query_endpoint():
    """
    Query endpoint for chatbot
    Request: { "text": "user query", "lang": "en" }
    Response: { "answer": "...", "source_id": 1, "score": 0.95, "detected_language": "en" }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        user_query = data['text'].strip()
        preferred_lang = data.get('lang', 'en')
        
        if not user_query:
            return jsonify({'error': 'Empty query'}), 400
        
        # Detect language
        detected_lang = detect_language(user_query)
        
        # Search for best match
        results = search_engine.search(user_query, top_k=1)
        
        if not results:
            return jsonify({
                'answer': 'I apologize, but I could not find a relevant answer to your question. Please try rephrasing or ask a different health-related question.',
                'source_id': None,
                'score': 0.0,
                'detected_language': detected_lang
            })
        
        best_faq, score = results[0]
        
        # Log query
        log_entry = QueryLog(
            user_query=user_query,
            detected_language=detected_lang,
            matched_faq_id=best_faq.id,
            confidence_score=score,
            ip_address=request.remote_addr
        )
        db.session.add(log_entry)
        db.session.commit()
        
        return jsonify({
            'answer': best_faq.answer,
            'source_id': best_faq.id,
            'score': round(score, 4),
            'detected_language': detected_lang,
            'question': best_faq.question
        })
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/import-faqs', methods=['POST'])
@limiter.limit("5 per hour")
def import_faqs():
    """
    Import FAQs from CSV file
    Request: { "file_path": "health_faqs_large.csv", "clear_existing": false }
    """
    try:
        data = request.get_json()
        file_path = data.get('file_path', 'health_faqs_large.csv')
        clear_existing = data.get('clear_existing', False)
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_path}'}), 404
        
        # Clear existing FAQs if requested
        if clear_existing:
            FAQ.query.delete()
            db.session.commit()
            logger.info("Cleared existing FAQs")
        
        # Import CSV
        imported_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                faq = FAQ(
                    question=row.get('question', '').strip(),
                    answer=row.get('answer', '').strip(),
                    language=row.get('language', 'en').strip()
                )
                
                if faq.question and faq.answer:
                    db.session.add(faq)
                    imported_count += 1
        
        db.session.commit()
        logger.info(f"Imported {imported_count} FAQs")
        
        # Rebuild search index
        load_faqs_from_db()
        
        return jsonify({
            'success': True,
            'imported': imported_count,
            'message': f'Successfully imported {imported_count} FAQs'
        })
        
    except Exception as e:
        logger.error(f"Import error: {e}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    total_faqs = FAQ.query.count()
    total_queries = QueryLog.query.count()
    
    return jsonify({
        'total_faqs': total_faqs,
        'total_queries': total_queries,
        'embeddings_enabled': search_engine.use_embeddings
    })

# ===== Database Initialization =====
def init_db():
    """Initialize database and load FAQs"""
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")
        
        # Load FAQs from database
        load_faqs_from_db()
        
        # If no FAQs, try to load from default CSV
        if len(search_engine.faqs) == 0:
            csv_path = 'health_faqs_large.csv'
            if os.path.exists(csv_path):
                logger.info(f"Loading FAQs from {csv_path}...")
                with app.test_request_context():
                    import_faqs_internal(csv_path)
            else:
                logger.warning(f"No FAQs found. Create {csv_path} and use /import-faqs endpoint")

def import_faqs_internal(file_path: str):
    """Internal function to import FAQs without HTTP context"""
    imported_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            faq = FAQ(
                question=row.get('question', '').strip(),
                answer=row.get('answer', '').strip(),
                language=row.get('language', 'en').strip()
            )
            if faq.question and faq.answer:
                db.session.add(faq)
                imported_count += 1
    
    db.session.commit()
    load_faqs_from_db()
    logger.info(f"Imported {imported_count} FAQs")

# ===== Error Handlers =====
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ===== Run Application =====
if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)