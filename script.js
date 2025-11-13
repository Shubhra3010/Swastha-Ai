//Handles chat interactions, API calls, and UI updates
 

// Configuration
const API_BASE_URL = 'http://localhost:5000';
const RATE_LIMIT_DELAY = 1000; // 1 second between requests

// State management
let lastRequestTime = 0;
let isProcessing = false;

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const languageSelector = document.getElementById('language-selector');
const loadingIndicator = document.getElementById('loading-indicator');

/**
 * Initialize the application
 */
function init() {
    // Check backend health on load
    checkBackendHealth();
    
    // Event listeners
    chatForm.addEventListener('submit', handleSubmit);
    userInput.addEventListener('input', autoResizeTextarea);
    
    // Example question buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            userInput.value = question;
            handleSubmit(new Event('submit'));
        });
    });
    
    // Language selector change
    languageSelector.addEventListener('change', () => {
        addSystemMessage(`Language changed to: ${languageSelector.options[languageSelector.selectedIndex].text}`);
    });
}

/**
 * Check if backend is healthy
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            showError('Backend service is not responding. Please check if the server is running.');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        showError('Cannot connect to backend. Please ensure the Flask server is running on http://localhost:5000');
    }
}

/**
 * Handle form submission
 */
async function handleSubmit(e) {
    e.preventDefault();
    
    const question = userInput.value.trim();
    if (!question || isProcessing) return;
    
    // Rate limiting check
    const now = Date.now();
    if (now - lastRequestTime < RATE_LIMIT_DELAY) {
        showError('Please wait a moment before sending another message.');
        return;
    }
    
    // Clear input and disable form
    userInput.value = '';
    setProcessing(true);
    
    // Remove welcome message if present
    const welcomeMsg = chatContainer.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();
    
    // Add user message to chat
    addMessage(question, 'user');
    
    // Scroll to bottom
    scrollToBottom();
    
    try {
        // Get selected language
        const language = languageSelector.value;
        
        // Call backend API
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: question,
                lang: language
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Request failed');
        }
        
        const data = await response.json();
        
        // Add bot response
        addMessage(
            data.answer,
            'bot',
            {
                score: data.score,
                sourceId: data.source_id,
                detectedLang: data.detected_language
            }
        );
        
        lastRequestTime = Date.now();
        
    } catch (error) {
        console.error('Query error:', error);
        showError(`Error: ${error.message}`);
    } finally {
        setProcessing(false);
        scrollToBottom();
    }
}

/**
 * Add message to chat
 */
function addMessage(text, type, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(contentDiv);
    
    // Add metadata for bot messages
    if (type === 'bot' && metadata.score !== undefined) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'confidence-score';
        scoreSpan.textContent = `${Math.round(metadata.score * 100)}% match`;
        
        const sourceSpan = document.createElement('span');
        sourceSpan.className = 'source-id';
        sourceSpan.textContent = ` • Source: ${metadata.sourceId}`;
        
        if (metadata.detectedLang) {
            const langSpan = document.createElement('span');
            langSpan.textContent = ` • Lang: ${metadata.detectedLang}`;
            metaDiv.appendChild(langSpan);
        }
        
        metaDiv.appendChild(scoreSpan);
        metaDiv.appendChild(sourceSpan);
        messageDiv.appendChild(metaDiv);
    }
    
    chatContainer.appendChild(messageDiv);
}

/**
 * Add system message
 */
function addSystemMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    messageDiv.style.textAlign = 'center';
    messageDiv.style.color = 'var(--text-secondary)';
    messageDiv.style.fontSize = '0.85rem';
    messageDiv.style.fontStyle = 'italic';
    messageDiv.textContent = text;
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message bot-message';
    errorDiv.style.color = '#dc2626';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.style.borderColor = '#dc2626';
    contentDiv.innerHTML = `⚠️ ${message}`;
    
    errorDiv.appendChild(contentDiv);
    chatContainer.appendChild(errorDiv);
    scrollToBottom();
}

/**
 * Set processing state
 */
function setProcessing(processing) {
    isProcessing = processing;
    sendBtn.disabled = processing;
    userInput.disabled = processing;
    
    if (processing) {
        loadingIndicator.style.display = 'flex';
    } else {
        loadingIndicator.style.display = 'none';
    }
}

/**
 * Auto-resize textarea based on content
 */
function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
}

/**
 * Handle Enter key (submit) vs Shift+Enter (new line)
 */
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
    }
});

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);