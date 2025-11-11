"""
Image Search Web Application
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from image_retrieval import ImageRetrieval
import os

app = Flask(__name__)

# 初始化检索系统
retrieval = None
INDEX_PATH = "image_index.pkl"


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    global retrieval
    
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'Please enter a search query'}), 400
    
    # Lazy load index
    if retrieval is None:
        if not os.path.exists(INDEX_PATH):
            return jsonify({'error': 'Index file not found. Please run build_index.py first'}), 404
        
        retrieval = ImageRetrieval()
        retrieval.load_index(INDEX_PATH)
    
    try:
        results = retrieval.search(query, top_k)
        
        # Format results
        formatted_results = [
            {
                'path': path,
                'score': score,
                'filename': os.path.basename(path)
            }
            for path, score in results
        ]
        
        return jsonify({
            'query': query,
            'results': formatted_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/image/<path:filename>')
def serve_image(filename):
    """Serve image files"""
    # Safely serve images
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    return send_from_directory(directory, basename)


if __name__ == '__main__':
    print("Starting Image Search Web Application...")
    print("For local access: http://127.0.0.1:6006")
    # app.run(debug=True, host='127.0.0.1', port=5000)
    print("For AutoDL: Check custom service link in console")
    app.run(debug=True, host='0.0.0.0', port=6006)


