from flask import Flask, request, jsonify, render_template
import g4f
import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import json

app = Flask(__name__)

# Available providers for comparison
PROVIDERS = [
    g4f.Provider.Bing,
    g4f.Provider.ChatgptAi,
    g4f.Provider.FreeGpt,
    g4f.Provider.GPTalk,
    g4f.Provider.Liaobots,
    g4f.Provider.OpenaiChat,
    g4f.Provider.Phind,
    g4f.Provider.Yqcloud,
    g4f.Provider.You,
    g4f.Provider.Aichat,
]

def test_provider(provider, prompt, model="gpt-3.5-turbo"):
    """Test a single provider with the given prompt"""
    start_time = time.time()
    result = {
        'provider': provider.__name__,
        'success': False,
        'response': '',
        'error': '',
        'response_time': 0,
        'model': model
    }
    
    try:
        response = g4f.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            timeout=30
        )
        
        result['success'] = True
        result['response'] = str(response)
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
    provider_list = [{'name': p.__name__, 'class': str(p)} for p in PROVIDERS]
    return jsonify(provider_list)

@app.route('/api/compare', methods=['POST'])
def compare_providers():
    """Compare multiple providers with a given prompt"""
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400
    
    prompt = data['prompt']
    selected_providers = data.get('providers', [])
    model = data.get('model', 'gpt-3.5-turbo')
    max_workers = data.get('max_workers', 5)
    
    # Filter providers based on selection
    if selected_providers:
        providers_to_test = [p for p in PROVIDERS if p.__name__ in selected_providers]
    else:
        providers_to_test = PROVIDERS
    
    results = []
    
    # Use ThreadPoolExecutor for concurrent testing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(test_provider, provider, prompt, model): provider 
            for provider in providers_to_test
        }
        
        for future in futures:
            try:
                result = future.result(timeout=35)
                results.append(result)
            except Exception as e:
                provider = futures[future]
                results.append({
                    'provider': provider.__name__,
                    'success': False,
                    'response': '',
                    'error': f'Timeout or execution error: {str(e)}',
                    'response_time': 0,
                    'model': model
                })
    
    # Sort results by success and response time
    results.sort(key=lambda x: (not x['success'], x['response_time']))
    
    return jsonify({
        'prompt': prompt,
        'model': model,
        'total_providers': len(results),
        'successful_providers': len([r for r in results if r['success']]),
        'results': results
    })

@app.route('/api/test-single', methods=['POST'])
def test_single_provider():
    """Test a single provider"""
    data = request.get_json()
    
    if not data or 'prompt' not in data or 'provider' not in data:
        return jsonify({'error': 'Prompt and provider are required'}), 400
    
    prompt = data['prompt']
    provider_name = data['provider']
    model = data.get('model', 'gpt-3.5-turbo')
    
    # Find the provider class
    provider = None
    for p in PROVIDERS:
        if p.__name__ == provider_name:
            provider = p
            break
    
    if not provider:
        return jsonify({'error': 'Provider not found'}), 404
    
    result = test_provider(provider, prompt, model)
    return jsonify(result)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
