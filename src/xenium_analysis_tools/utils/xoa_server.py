import http.server
import socketserver
import threading
import socket
from pathlib import Path

def find_free_port(start_port=8000, max_port=8100):
    """Find a free port starting from start_port"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found between {start_port} and {max_port}")

def start_server(directory, port=None):
    """Start a simple HTTP server in the background"""
    import os
    original_dir = os.getcwd()
    
    try:
        # Find a free port if none specified
        if port is None:
            port = find_free_port()
        
        os.chdir(directory)
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        
        # Start server in background thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        print(f"Server started at http://localhost:{port}")
        # print(f"Access your file at: http://localhost:{port}/{Path(html_name).name}")
        # print(f"Browse all files at: http://localhost:{port}")
        
        return httpd
    except Exception as e:
        print(f"Server start failed: {e}")
        os.chdir(original_dir)
        return None

def stop_server(httpd):
    """Stop the HTTP server"""
    if httpd:
        httpd.shutdown()
        print("Server stopped.")