from flask import Flask, request, jsonify, render_template
import time
import traceback
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import requests
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Mock providers for testing (replace with actual g4f when it works)
PROVIDERS = [
    {'name': 'OpenAI_GPT35', 'url': 'https://api.openai.com/v1/chat/completions'},
    {'name': 'Anthropic_Claude', 'url': 'https://api.anthropic.com/v1/messages'},
    {'name': 'Google_Bard', 'url': 'https://generativelanguage.googleapis.com/v1/models'},
    {'name': 'Cohere_Command', 'url': 'https://api.cohere.ai/v1/generate'},
    {'name': 'Hugging_Face', 'url': 'https://api-inference.huggingface.co/models'},
    {'name': 'Replicate_AI', 'url': 'https://api.replicate.com/v1/predictions'},
]

# Try to import g4f, but continue without it if it fails
try:
    import g4f
    G4F_AVAILABLE = True
    logger.info("g4f imported successfully")
    
    # Update providers with actual g4f providers
    G4F_PROVIDERS = [
        g4f.Provider.Bing,
        g4f.Provider.You,
        g4f.Provider.Aichat,
        g4f.Provider.ChatBase,
        g4f.Provider.Vercel,
    ]
except ImportError as e:
    G4F_AVAILABLE = False
    G4F_PROVIDERS = []
    logger.warning(f"g4f not available: {e}")
except Exception as e:
    G4F_AVAILABLE = False
    G4F_PROVIDERS = []
    logger.warning(f"g4f initialization failed: {e}")

def test_g4f_provider(provider, prompt, model="gpt-3.5-turbo"):
    """Test a g4f provider with the given prompt"""
    start_time = time.time()
    result = {
        'provider': provider.__name__,
        'success': False,
        'response': '',
        'error': '',
        'response_time': 0,
        'model': model,
        'type': 'g4f'
    }
    
    try:
        response = g4f.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            timeout=20
        )
        
        result['success'] = True
        result['response'] = str(response)
        result['response_time'] = round(time.time() - start_time, 2)
        
    except Exception as e:
        result['error'] = str(e)
        result['response_time'] = round(time.time() - start_time, 2)
    
    return result

def test_mock_provider(provider, prompt, model="gpt-3.5-turbo"):
    """Test a mock provider (for demo purposes)"""
    start_time = time.time()
    result = {
        'provider': provider['name'],
        'success': False,
        'response': '',
        'error': '',
        'response_time': 0,
        'model': model,
        'type': 'mock'
    }
    
    try:
        # Simulate different response times and success rates
        import random
        time.sleep(random.uniform(0.5, 3.0))  # Simulate processing time
        
        # 70% success rate for demo
        if random.random() > 0.3:
            mock_responses = [
                f"This is a response from {provider['name']} to your prompt: '{prompt[:50]}...' using model {model}.",
                f"According to {provider['name']}, here's what I found about your query: {prompt[:30]}...",
                f"{provider['name']} suggests that your question about '{prompt[:40]}...' requires careful consideration.",
                f"From {provider['name']}: Your prompt '{prompt[:35]}...' is interesting. Let me provide some insights.",
                f"Response from {provider['name']}: Based on your request '{prompt[:45]}...', here's my analysis."
            ]
            result['success'] = True
            result['response'] = random.choice(mock_responses)
        else:
            errors = [
                "Rate limit exceeded",
                "Provider temporarily unavailable",
                "Connection timeout",
                "Authentication failed",
                "Service overloaded"
            ]
            result['error'] = random.choice(errors)
        
        result['response_time'] = round(time.time() - start_time, 2)
        
    except Exception as e:
        result['error'] = str(e)
        result['response_time'] = round(time.time() - start_time, 2)
    
    return result

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get list of available providers"""
    provider_list = []
    
    # Add g4f providers if available
    if G4F_AVAILABLE:
        for p in G4F_PROVIDERS:
            provider_list.append({
                'name': p.__name__,
                'type': 'g4f',
                'status': 'available'
            })
    
    # Add mock providers for demo
    for p in PROVIDERS:
        provider_list.append({
            'name': p['name'],
            'type': 'mock',
            'status': 'demo'
        })
    
    return jsonify(provider_list)

@app.route('/api/compare', methods=['POST'])
def compare_providers():
    """Compare multiple providers with a given prompt"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        selected_providers = data.get('providers', [])
        model = data.get('model', 'gpt-3.5-turbo')
        max_workers = min(data.get('max_workers', 3), 5)  # Limit concurrent workers
        
        logger.info(f"Comparing providers for prompt: {prompt[:50]}...")
        
        # Determine which providers to test
        providers_to_test = []
        
        if selected_providers:
            # Test selected providers
            if G4F_AVAILABLE:
                for p in G4F_PROVIDERS:
                    if p.__name__ in selected_providers:
                        providers_to_test.append(('g4f', p))
            
            for p in PROVIDERS:
                if p['name'] in selected_providers:
                    providers_to_test.append(('mock', p))
        else:
            # Test all providers
            if G4F_AVAILABLE:
                providers_to_test.extend([('g4f', p) for p in G4F_PROVIDERS])
            providers_to_test.extend([('mock', p) for p in PROVIDERS])
        
        results = []
        
        # Use ThreadPoolExecutor for concurrent testing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for provider_type, provider in providers_to_test:
                if provider_type == 'g4f':
                    future = executor.submit(test_g4f_provider, provider, prompt, model)
                else:
                    future = executor.submit(test_mock_provider, provider, prompt, model)
                futures.append((future, provider_type, provider))
            
            for future, provider_type, provider in futures:
                try:
                    result = future.result(timeout=25)
                    results.append(result)
                    logger.info(f"Completed test for {result['provider']}: success={result['success']}")
                except Exception as e:
                    provider_name = provider.__name__ if provider_type == 'g4f' else provider['name']
                    results.append({
                        'provider': provider_name,
                        'success': False,
                        'response': '',
                        'error': f'Execution error: {str(e)}',
                        'response_time': 0,
                        'model': model,
                        'type': provider_type
                    })
                    logger.error(f"Error testing {provider_name}: {e}")
        
        # Sort results by success and response time
        results.sort(key=lambda x: (not x['success'], x['response_time']))
        
        successful_count = len([r for r in results if r['success']])
        
        logger.info(f"Comparison complete: {successful_count}/{len(results)} providers successful")
        
        return jsonify({
            'prompt': prompt,
            'model': model,
            'total_providers': len(results),
            'successful_providers': successful_count,
            'g4f_available': G4F_AVAILABLE,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in compare_providers: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/test-single', methods=['POST'])
def test_single_provider():
    """Test a single provider"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data or 'provider' not in data:
            return jsonify({'error': 'Prompt and provider are required'}), 400
        
        prompt = data['prompt']
        provider_name = data['provider']
        model = data.get('model', 'gpt-3.5-turbo')
        
        logger.info(f"Testing single provider: {provider_name}")
        
        # Find the provider
        result = None
        
        # Check g4f providers
        if G4F_AVAILABLE:
            for p in G4F_PROVIDERS:
                if p.__name__ == provider_name:
                    result = test_g4f_provider(p, prompt, model)
                    break
        
        # Check mock providers
        if not result:
            for p in PROVIDERS:
                if p['name'] == provider_name:
                    result = test_mock_provider(p, prompt, model)
                    break
        
        if not result:
            return jsonify({'error': 'Provider not found'}), 404
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in test_single_provider: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'g4f_available': G4F_AVAILABLE,
        'timestamp': time.time()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
