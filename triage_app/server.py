import http.server
import socketserver
import json
import os
import sys
import time
from urllib.parse import urlparse

# Ensure current dir is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from triage_engine import MultimodalTriageEngine
from presets import PRESET_SCENARIOS

WEB_DIR = os.path.join(CURRENT_DIR, "web")
os.makedirs(WEB_DIR, exist_ok=True)

engine = MultimodalTriageEngine()

class TriageRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def log_message(self, format, *args):
        # Keep terminal focused by ignoring routine static assets (.css, .js, .svg, etc.)
        if any(self.path.endswith(ext) for ext in [".css", ".js", ".png", ".jpg", ".webp", ".ico", ".woff2", ".ttf"]):
            return
        now_str = time.strftime('%H:%M:%S')
        code = args[1] if len(args) > 1 else '200'
        print(f"[{now_str}] [HTTP {code}] {self.command} {self.path}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(PRESET_SCENARIOS).encode("utf-8"))
        elif parsed.path == "/api/system_status":
            status = {
                "system": "Edge-Powered Multimodal Emergency Triage System",
                "team": "Team 10 - CSE220 Multimodal AI Systems",
                "institution": "Kalasalingam Academy of Research and Education",
                "models": {
                    "vision": "Google ViT (google/vit-base-patch16-224)",
                    "nlp": "DistilBERT (distilbert-base-uncased)",
                    "classifier": "Balanced Random Forest (1568-D Early Fusion)"
                },
                "status": "ONLINE",
                "edge_device": "Local Accelerated Edge Node"
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/triage":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_body.decode("utf-8"))
                text = data.get("text", "")
                category = data.get("category", "General Emergency")
                iot_data = data.get("iot_data", {})
                image_b64 = data.get("image_b64", None)
                override_hazard = data.get("image_hazard_override", None)
                active_modalities = data.get("active_modalities", {"visual": True, "textual": True, "iot": True})
                
                result = engine.predict_triage(
                    text=text,
                    category=category,
                    image_b64=image_b64,
                    iot_data=iot_data,
                    image_hazard_override=override_hazard,
                    active_modalities=active_modalities
                )
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port=8080):
    handler = TriageRequestHandler
    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True
    for p in [port, 8081, 8085, 5000]:
        try:
            with socketserver.TCPServer(("", p), handler) as httpd:
                print(f"\n=======================================================")
                print(f" EDGE MULTIMODAL TRIAGE COMMAND CENTER IS RUNNING!")
                print(f" Web Interface: http://localhost:{p}")
                print(f"=======================================================\n")
                httpd.serve_forever()
                break
        except OSError:
            print(f"Port {p} in use, trying next...")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
