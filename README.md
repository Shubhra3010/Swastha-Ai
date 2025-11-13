# Swasth AI - System Architecture & Integration

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  ┌───────────────┐              ┌────────────────┐         │
│  │  index.html   │              │   admin.html   │         │
│  │  (Main Chat)  │              │  (Admin Panel) │         │
│  └───────┬───────┘              └────────┬───────┘         │
│          │                               │                  │
│          └───────────┬───────────────────┘                  │
│                      │                                       │
│              ┌───────▼────────┐                             │
│              │   script.js    │                             │
│              │  (Frontend)    │                             │
│              └───────┬────────┘                             │
│                      │ HTTP/JSON                            │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       │ CORS-enabled REST API
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  FLASK BACKEND (app.py)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Endpoints:                                       │  │
│  │  • GET  /health      → Health check                  │  │
│  │  • POST /query       → Semantic search               │  │
│  │  • POST /import-faqs → Load CSV data                 │  │
│  │  • GET  /stats       → System statistics             │  │
│  └──────────────┬───────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────────┐  │
│  │        SEMANTIC SEARCH ENGINE                        │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  Sentence Transformers (Primary)             │   │  │
│  │  │  • Model: paraphrase-multilingual-MiniLM    │   │  │
│  │  │  • Embedding-based similarity search        │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │  TF-IDF Vectorizer (Fallback)                │   │  │
│  │  │  • Scikit-learn implementation               │   │  │
│  │  │  • Cosine similarity matching                │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  └──────────────┬───────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────────┐  │
│  │         LANGUAGE DETECTION                           │  │
│  │  • langdetect library                                │  │
│  │  • Auto-detect user query language                   │  │
│  └──────────────┬───────────────────────────────────────┘  │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────────┐  │
│  │       DATABASE LAYER (SQLAlchemy ORM)                │  │
│  │  ┌────────────┐          ┌─────────────┐            │  │
│  │  │ FAQ Table  │          │ QueryLog    │            │  │
│  │  │ (Questions)│          │ (Analytics) │            │  │
│  │  └────────────┘          └─────────────┘            │  │
│  └──────────────┬───────────────────────────────────────┘  │
│                 │                                            │
└─────────────────┼────────────────────────────────────────────┘
                  │
         ┌────────▼─────────┐
         │  SQLite Database │
         │  swasth_ai.db    │
         └──────────────────┘
```

## 🔄 Data Flow

### 1. User Query Flow

```
User enters question
      ↓
Frontend (script.js) captures input
      ↓
AJAX POST request to /query endpoint
      ↓
Backend receives { text: "question", lang: "en" }
      ↓
Language Detection (langdetect)
      ↓
Semantic Search Engine
      ├─→ Generate embeddings (if available)
      └─→ TF-IDF vectorization (fallback)
      ↓
Compute similarity scores with FAQ database
      ↓
Retrieve best matching FAQ
      ↓
Log query to QueryLog table
      ↓
Return { answer, source_id, score, detected_language }
      ↓
Frontend displays response in chat UI
```

### 2. FAQ Import Flow

```
Admin opens admin.html
      ↓
Enters CSV file path
      ↓
POST request to /import-faqs endpoint
      ↓
Backend reads CSV file
      ↓
Parse rows into FAQ objects
      ↓
Save to database (SQLite)
      ↓
Rebuild search index
      ├─→ Generate embeddings for all FAQs
      └─→ Build TF-IDF matrix
      ↓
Return success/failure response
      ↓
Admin panel shows confirmation
```

## 🧩 Component Integration

### Frontend ↔ Backend Communication

**index.html + script.js**
- Uses `fetch()` API for HTTP requests
- Sends JSON payloads
- Handles CORS automatically (enabled on backend)
- Implements client-side rate limiting
- Displays loading states

**Backend (app.py)**
- Flask-CORS enables cross-origin requests
- JSON request/response format
- Flask-Limiter provides server-side rate limiting
- Returns standardized error messages

### Search Engine Integration

**Initialization:**
```python
search_engine = SemanticSearchEngine()
# Automatically selects embeddings or TF-IDF
```

**Building Index:**
```python
faqs = FAQ.query.all()
search_engine.build_index(faqs)
# Generates embeddings or TF-IDF vectors
```

**Searching:**
```python
results = search_engine.search(query, top_k=1)
# Returns [(faq, score)] list
```

### Database Schema

**FAQ Table:**
```sql
CREATE TABLE faqs (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**QueryLog Table:**
```sql
CREATE TABLE query_logs (
    id INTEGER PRIMARY KEY,
    user_query TEXT NOT NULL,
    detected_language VARCHAR(10),
    matched_faq_id INTEGER,
    confidence_score FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    FOREIGN KEY (matched_faq_id) REFERENCES faqs(id)
);
```

## 🎯 Key Integration Points

### 1. Frontend → Backend API Calls

**Health Check:**
```javascript
fetch('http://localhost:5000/health')
  .then(res => res.json())
  .then(data => console.log(data.status));
```

**Query:**
```javascript
fetch('http://localhost:5000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: query, lang: 'en' })
})
  .then(res => res.json())
  .then(data => displayAnswer(data));
```

### 2. Database → Search Engine

```python
# Load FAQs from database
faqs = FAQ.query.all()

# Build search index
texts = [f"{faq.question} {faq.answer}" for faq in faqs]
embeddings = model.encode(texts)

# Search
query_embedding = model.encode([user_query])[0]
similarities = np.dot(embeddings, query_embedding)
best_match_idx = np.argmax(similarities)
```

### 3. CSV Import → Database

```python
with open('health_faqs_large.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        faq = FAQ(
            question=row['question'],
            answer=row['answer'],
            language=row['language']
        )
        db.session.add(faq)
db.session.commit()
```

## 🔐 Security Features

### Rate Limiting
```python
@limiter.limit("10 per minute")
def query_endpoint():
    # Prevents abuse
```

### CORS Protection
```python
CORS(app, resources={r"/*": {"origins": "*"}})
# Configure specific origins in production
```

### Input Validation
```python
if not data or 'text' not in data:
    return jsonify({'error': 'Missing field'}), 400
```

## 📊 Monitoring & Analytics

### Query Logging
Every query is logged with:
- User query text
- Detected language
- Matched FAQ ID
- Confidence score
- Timestamp
- IP address

### Statistics Endpoint
```json
{
  "total_faqs": 25,
  "total_queries": 150,
  "embeddings_enabled": true
}
```

## 🚀 Deployment Considerations

### Development Setup
```bash
# Backend
python app.py  # localhost:5000

# Frontend
python -m http.server 8000  # localhost:8000
```

### Production Setup

**Backend (Gunicorn + Nginx):**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Frontend (Nginx static):**
```nginx
server {
    listen 80;
    root /var/www/swasth-ai;
    index index.html;
}
```

**Database Migration:**
```python
# Switch to PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 
    'postgresql://user:pass@localhost/swasth_ai'
```

## 🧪 Testing Integration

### Unit Tests (test_app.py)
- Test each API endpoint
- Mock database with in-memory SQLite
- Verify semantic search functionality
- Check query logging

### Run Tests:
```bash
pytest test_app.py -v
```

## 📈 Performance Optimization

### Caching Strategy
```python
# Add Redis caching for frequent queries
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@cache.memoize(timeout=300)
def search_faqs(query):
    return search_engine.search(query)
```

### Database Indexing
```sql
CREATE INDEX idx_faq_language ON faqs(language);
CREATE INDEX idx_query_timestamp ON query_logs(timestamp);
```

### Batch Processing
```python
# Build embeddings in batches
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings[i:i+batch_size] = model.encode(batch)
```

## 🌐 Multilingual Support

### Language Detection
```python
from langdetect import detect
detected_lang = detect(user_query)  # Returns 'en', 'hi', etc.
```

### Language-Specific Search
```python
# Filter FAQs by language
faqs = FAQ.query.filter_by(language=detected_lang).all()
```

### Translation (Future Enhancement)
```python
# Use Google Translate API or similar
from googletrans import Translator
translator = Translator()
translated = translator.translate(query, dest='en')
```

## 🔧 Troubleshooting Integration Issues

### Frontend can't reach backend
- Check CORS configuration
- Verify backend is running on correct port
- Check browser console for errors

### Embeddings not loading
- System automatically falls back to TF-IDF
- Check if model files downloaded (~500MB)
- Verify `sentence-transformers` installed

### Database connection issues
- Check SQLite file permissions
- Verify database initialized: `db.create_all()`
- Check for locked database file

### Search returns no results
- Verify FAQs loaded: `FAQ.query.count()`
- Check search index built: `search_engine.faqs`
- Ensure query not empty

---

This architecture provides a solid foundation for a production-ready multilingual health chatbot with room for scaling and enhancement.
