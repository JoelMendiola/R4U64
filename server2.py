#!/usr/bin/env python3
"""Servidor estatico para la demo Handspace."""
import http.server
import os
import socketserver
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

with socketserver.ThreadingTCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Sirviendo Handspace en http://localhost:{PORT}  (Ctrl+C para detener)")
    httpd.serve_forever()
