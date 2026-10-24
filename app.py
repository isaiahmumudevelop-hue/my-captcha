from flask import Flask, render_template_string, send_file, request, jsonify, session
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Prevents massive image crashes
import io
import uuid
import os
import random
import cv2
import numpy as np
import time
import gc # Garbage Collector to free up RAM
from gtts import gTTS

app = Flask(__name__)
app.secret_key = 'super_secret_captcha_key_123'

print("Loading YOLO...")
net_yolo = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net_yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in net_yolo.getUnconnectedOutLayers()]

print("Loading MobileNet...")
net_mobilenet = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
print("All AI Models Loaded Successfully!")

captcha_data = {}
issued_tokens = {} # Ray ID storage

# --- ADMIN DASHBOARD STATS ---
stats = {
    "total_loads": 0, 
    "success": 0, 
    "failures": 0,
    "level1_success": 0,
    "level2_success": 0
}

# --- YOLO TARGETS (3x3) ---
YOLO_TARGETS = {
    2: {"name": "cars", "folder": "real_images/cars", "icon": "🚗"},
    6: {"name": "trains", "folder": "real_images/trains", "icon": "🚆"},
    1: {"name": "bicycles", "folder": "real_images/bicycles", "icon": "🚲"},
    5: {"name": "buses", "folder": "real_images/buses", "icon": "🚌"},
    0: {"name": "people", "folder": "real_images/people", "icon": "🚶"},
    9: {"name": "traffic lights", "folder": "real_images/traffic_lights", "icon": "🚦"}
}

# --- MOBILENET TARGETS (4x4) ---
MOBILENET_TARGETS = {
    7: {"name": "cars", "folder": "real_images/cars", "icon": "🚗"},
    6: {"name": "buses", "folder": "real_images/buses", "icon": "🚌"},
    14: {"name": "motorcycles", "folder": "real_images/motorcycles", "icon": "🏍️"},
    2: {"name": "bicycles", "folder": "real_images/bicycles", "icon": "🚲"}
}

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Interactive CAPTCHA</title>
    <style>
        body { font-family: 'Roboto', sans-serif; padding: 20px; background: #f0f0f0; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; transition: background 0.3s; }
        
        /* --- DARK MODE STYLES --- */
        body.dark-mode { background: #121212; }
        body.dark-mode #theme-toggle { background: #333; color: white; border-color: #555; }
        body.dark-mode #initial-checkbox { background: #2d2d2d; border-color: #444; }
        body.dark-mode #initial-checkbox:hover { background: #383838; }
        body.dark-mode .cb-text { color: white; }
        body.dark-mode .cb-terms a { color: #aaa; }
        body.dark-mode .captcha-box { background: #1e1e1e; box-shadow: 0 2px 10px rgba(255,255,255,0.1); }
        body.dark-mode .content { background: #1e1e1e; }
        body.dark-mode .grid-3, body.dark-mode .grid-4 { background: #333; }
        body.dark-mode .verify-btn { background: #1a73e8; }
        body.dark-mode .icons button { color: #ccc; }
        body.dark-mode .icons button:hover { color: #1a73e8; }
        body.dark-mode #success-area p { color: white; }
        body.dark-mode .legal-content { background: #1e1e1e; color: white; }
        body.dark-mode .legal-content h3, body.dark-mode .legal-content h4 { color: #66b2ff; }
        body.dark-mode .legal-content p { color: #ccc; }
        body.dark-mode .btn-close { background: #333; color: white; }
        body.dark-mode #audio-modal { background: rgba(0,0,0,0.95); }
        body.dark-mode #audio-modal h4, body.dark-mode #audio-modal p { color: white; }
        body.dark-mode #timer-display { color: #ff6b6b; }
        body.dark-mode #ray-id-container { background: #2d2d2d; border: 1px solid #444; }
        body.dark-mode #ray-id { color: #66b2ff; }

        .captcha-box { background: white; border-radius: 4px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); width: 330px; overflow: hidden; position: relative; display: none; }
        
        #initial-checkbox { 
            display: none; justify-content: space-between; align-items: center; 
            padding: 20px; background: #f9f9f9; border: 1px solid #d3d3d3; 
            border-radius: 3px; width: 300px; margin: 0 auto; cursor: pointer;
            box-shadow: 0 0 4px 1px rgba(0,0,0,0.08); transition: background 0.2s;
        }
        #initial-checkbox:hover { background: #ffffff; }
        .checkbox-left { display: flex; align-items: center; gap: 12px; }
        .cb-box { width: 28px; height: 28px; border: 2px solid #c1c1c1; border-radius: 2px; background: #fff; }
        .cb-text { font-size: 14px; color: #000; font-weight: 500; }
        .checkbox-right { text-align: center; display: flex; flex-direction: column; align-items: center; }
        .cb-brand { font-size: 10px; color: #555; margin-top: 2px; }
        .cb-terms { font-size: 8px; color: #555; }
        
        .cb-terms a { color: #555; text-decoration: none; }
        .cb-terms a:hover { text-decoration: underline; }

        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; display: none; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .header { background: #1a73e8; color: white; padding: 15px; display: flex; justify-content: space-between; align-items: center; }
        .header-text { font-size: 16px; font-weight: bold; line-height: 1.2; }
        .header-image { width: 60px; height: 60px; background: #fff; border: 2px solid white; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 30px;}
        .content { padding: 15px; }
        
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; margin-bottom: 15px; width: 300px; height: 300px; background: #ccc; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; margin-bottom: 15px; width: 300px; height: 300px; background: #ccc; }
        
        .tile { cursor: pointer; position: relative; box-sizing: border-box; }
        .tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
        
        .tile.selected::after {
            content: '✓'; position: absolute; top: 4px; left: 4px; color: white; font-weight: bold;
            font-size: 14px; background: #1a73e8; width: 20px; height: 20px;
            border-radius: 50%; display: flex; align-items: center; justify-content: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        }
        .tile.selected { border: 3px solid #1a73e8; }
        .tile.selected img { opacity: 0.6; }
        
        .controls { display: flex; justify-content: space-between; align-items: center; }
        .icons { display: flex; gap: 15px; font-size: 20px; color: #555; }
        .icons button { background: none; border: none; font-size: 20px; cursor: pointer; color: #555; padding: 0; transition: 0.2s; }
        .icons button:hover { color: #1a73e8; transform: scale(1.1); }
        
        .verify-btn { background: #1a73e8; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px; transition: 0.2s; }
        .verify-btn:hover { background: #1557b0; }
        .verify-btn:disabled { background: #999; cursor: not-allowed; }

        .skip-btn { background: #f44336; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px; margin-right: 10px; display: none; }
        .skip-btn:hover { background: #d32f2f; }
        
        #success-area { display: none; text-align: center; padding: 40px 0; }
        #success-area h2 { color: #4CAF50; margin: 0 0 10px 0; font-size: 24px; }
        #success-area p { color: #333; font-size: 16px; margin: 0; }

        #audio-modal { display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.95); z-index: 10; padding: 20px; box-sizing: border-box; flex-direction: column; justify-content: center; align-items: center; }
        #audio-modal h4 { margin-top: 0; color: #333; }
        #audio-modal input { padding: 10px; font-size: 16px; width: 150px; margin: 15px 0; border: 1px solid #ccc; border-radius: 4px; text-align: center; text-transform: uppercase; }
        #audio-modal .audio-controls { display: flex; gap: 10px; }
        #audio-modal button { padding: 8px 15px; cursor: pointer; border-radius: 4px; border: none; font-weight: bold; }
        .btn-play { background: #1a73e8; color: white; }
        .btn-close { background: #eee; color: #333; }

        #legal-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 1000; justify-content: center; align-items: center; }
        .legal-content { background: white; padding: 30px; border-radius: 8px; width: 80%; max-width: 500px; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .legal-content h3 { margin-top: 0; color: #1a73e8; }
        .legal-content h4 { margin-bottom: 5px; color: #333; }
        .legal-content p { font-size: 13px; color: #555; line-height: 1.5; margin-bottom: 15px; }
        .legal-close { background: #1a73e8; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; float: right; font-weight: bold; }
    </style>
</head>
<body>

    <!-- DARK MODE TOGGLE BUTTON -->
    <button id="theme-toggle" onclick="toggleTheme()" style="position: fixed; top: 20px; right: 20px; padding: 10px 15px; border-radius: 20px; border: 1px solid #ccc; background: white; cursor: pointer; font-size: 16px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1); z-index: 1001; transition: 0.3s;">🌙 Dark</button>

    <!-- INITIAL CHECKBOX -->
    <div id="initial-checkbox" onclick="startVerification()">
        <div class="checkbox-left">
            <div id="cb-box" class="cb-box"></div>
            <div id="cb-spinner" class="spinner"></div>
            <svg id="cb-checkmark" style="display:none; width: 28px; height: 28px;" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 6L9 17L4 12" stroke="#4CAF50" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span class="cb-text">I'm not a robot</span>
        </div>
        <div class="checkbox-right">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12" stroke="#1a73e8" stroke-width="2" stroke-linecap="round"/>
                <path d="M12 6V12L16 14" stroke="#1a73e8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M22 12H18" stroke="#1a73e8" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <div class="cb-brand">AI CAPTCHA</div>
            <div class="cb-terms"><a href="#" onclick="event.stopPropagation(); openLegal('privacy')">Privacy</a> - <a href="#" onclick="event.stopPropagation(); openLegal('terms')">Terms</a></div>
        </div>
    </div>

    <!-- MAIN CAPTCHA BOX -->
    <div class="captcha-box" id="main-captcha-box">
        <div class="header">
            <div class="header-text" id="challenge-text">Select all images with<br><span style="font-size:20px;">{{ challenge_target }}</span></div>
            <div class="header-image">{{ target_icon }}</div>
        </div>
        <div class="content" id="main-content">
            <div id="timer-display" style="font-size: 12px; color: #d32f2f; text-align: right; margin-bottom: 5px; font-weight: bold; display: none;">Time remaining: 2:00</div>
            <div class="{{ grid_class }}" id="captcha-grid"></div>
            <div class="controls" id="controls-area">
                <div class="icons">
                    <button type="button" onclick="refreshCaptcha()" title="Get new challenge">↻</button>
                    <button type="button" onclick="openAudio()" title="Audio Challenge">🎧</button>
                    <button type="button" title="Help">ℹ️</button>
                </div>
                <div>
                    <button type="button" class="skip-btn" id="skip-btn" onclick="skipChallenge()">Skip</button>
                    <button type="button" class="verify-btn" id="verify-btn" onclick="verifyCaptcha()">VERIFY</button>
                </div>
            </div>
        </div>
        
        <div class="content" id="success-area">
            <h2>✔ Success!</h2>
            <p>You are not a robot.</p>
            <div id="ray-id-container" style="display: none; margin-top: 15px; padding: 10px; background: #e8f0fe; border-radius: 4px; word-break: break-all;">
                <p style="font-size: 10px; color: #555; margin: 0 0 5px 0;">Ray 🆔 (Valid for 5 mins):</p>
                <code id="ray-id" style="font-size: 12px; font-weight: bold; color: #1a73e8;"></code>
            </div>
        </div>

        <div id="audio-modal">
            <h4>Audio Challenge</h4>
            <p>Listen to the numbers and type them below.</p>
            <audio id="audio-player" src="/audio/{{ session_id }}" preload="auto"></audio>
            <button type="button" class="btn-play" onclick="document.getElementById('audio-player').play()">▶ Play Audio</button>
            <input type="text" id="audio-input" placeholder="Enter code" maxlength="6">
            <div class="audio-controls">
                <button type="button" class="btn-close" onclick="closeAudio()">Cancel</button>
                <button type="button" class="btn-play" onclick="verifyAudio()">Submit</button>
            </div>
        </div>
    </div>

    <!-- LEGAL MODAL -->
    <div id="legal-modal">
        <div class="legal-content">
            <button class="legal-close" onclick="closeLegal()">Close</button>
            <div id="legal-text"></div>
        </div>
    </div>

    <script>
        let selectedTiles = [];
        let timerInterval;
        let timeLeft = 120;
        const sessionId = "{{ session_id }}";
        const totalTiles = {{ total_tiles }};
        const showSkip = {{ show_skip | tojson }};
        const showCheckbox = {{ show_checkbox | tojson }};
        let isVerified = false;

        const legalData = {
            privacy: `
                <h3>Privacy Policy</h3>
                <p><strong>Last Updated: Today</strong></p>
                <h4>1. Data Processing</h4>
                <p>This CAPTCHA is designed with privacy in mind. All image processing and AI object detection occurs locally on the server hosting this page. We do not transmit your interaction data to third-party advertising networks.</p>
                <h4>2. Cookies & Sessions</h4>
                <p>We use temporary session tokens to track the progression of your CAPTCHA challenge. These tokens are deleted immediately upon successful or failed verification and do not contain personally identifiable information.</p>
                <h4>3. Audio Challenge</h4>
                <p>If you use the audio challenge, a temporary audio file is generated using Text-to-Speech. This audio is generated in memory and is not stored or linked to your identity.</p>
            `,
            terms: `
                <h3>Terms of Service</h3>
                <p><strong>Last Updated: Today</strong></p>
                <h4>1. Purpose</h4>
                <p>This system is implemented to prevent automated bots, scrapers, and malicious scripts from accessing protected resources. By using this service, you agree to interact with it as a human user.</p>
                <h4>2. Acceptable Use</h4>
                <p>You agree not to use automated scripts, AI solvers, or macros to bypass this CAPTCHA. Any attempt to reverse-engineer the AI models or bypass the security measures is strictly prohibited.</p>
                <h4>3. Liability</h4>
                <p>This CAPTCHA is provided "as is". We are not responsible for any delays, errors, or failure to grant access if you are unable to complete the visual or audio challenges correctly.</p>
            `
        };

        window.onload = () => {
            if (showCheckbox) {
                document.getElementById('initial-checkbox').style.display = 'flex';
            } else {
                document.getElementById('main-captcha-box').style.display = 'block';
                loadGrid();
            }
        };

        function toggleTheme() {
            document.body.classList.toggle('dark-mode');
            const btn = document.getElementById('theme-toggle');
            if (document.body.classList.contains('dark-mode')) {
                btn.innerText = '☀️ Light';
            } else {
                btn.innerText = '🌙 Dark';
            }
        }

        function openLegal(type) {
            document.getElementById('legal-text').innerHTML = legalData[type];
            document.getElementById('legal-modal').style.display = 'flex';
        }

        function closeLegal() {
            document.getElementById('legal-modal').style.display = 'none';
        }

        function startVerification() {
            if (isVerified) return;
            document.getElementById('cb-box').style.display = 'none';
            document.getElementById('cb-spinner').style.display = 'block';
            
            setTimeout(() => {
                document.getElementById('cb-spinner').style.display = 'none';
                document.getElementById('cb-checkmark').style.display = 'block';
                
                setTimeout(() => {
                    isVerified = true;
                    document.getElementById('initial-checkbox').style.display = 'none';
                    document.getElementById('main-captcha-box').style.display = 'block';
                    loadGrid();
                }, 800);
            }, 1000);
        }

        function startTimer() {
            if (timerInterval) clearInterval(timerInterval);
            timeLeft = 120;
            
            timerInterval = setInterval(() => {
                timeLeft--;
                let minutes = Math.floor(timeLeft / 60);
                let seconds = timeLeft % 60;
                if (seconds < 10) seconds = "0" + seconds;
                
                document.getElementById('timer-display').innerText = `Time remaining: ${minutes}:${seconds}`;
                
                if (timeLeft <= 0) {
                    clearInterval(timerInterval);
                    alert("Time expired! Please try again.");
                    window.location.reload();
                }
            }, 1000);
        }

        function loadGrid() {
            if (showSkip) {
                document.getElementById('skip-btn').style.display = 'block';
            }
            document.getElementById('timer-display').style.display = 'block';
            startTimer();
            
            const grid = document.getElementById('captcha-grid');
            for (let i = 0; i < totalTiles; i++) {
                const tile = document.createElement('div');
                tile.className = 'tile';
                tile.dataset.index = i;
                const img = document.createElement('img');
                img.src = `/tile/${sessionId}/${i}`;
                tile.appendChild(img);
                tile.onclick = () => {
                    tile.classList.toggle('selected');
                    if (tile.classList.contains('selected')) {
                        selectedTiles.push(i);
                    } else {
                        selectedTiles = selectedTiles.filter(idx => idx !== i);
                    }
                };
                grid.appendChild(tile);
            }
        }

        function refreshCaptcha() { window.location.reload(); }
        function skipChallenge() { fetch('/skip', { method: 'POST' }).then(() => window.location.reload()); }

        function openAudio() {
            document.getElementById('audio-modal').style.display = 'flex';
            document.getElementById('audio-player').play();
        }

        function closeAudio() {
            document.getElementById('audio-modal').style.display = 'none';
            document.getElementById('audio-input').value = '';
        }

        function verifyAudio() {
            const code = document.getElementById('audio-input').value;
            fetch('/verify_audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, code: code })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    closeAudio();
                    clearInterval(timerInterval);
                    document.getElementById('main-content').style.display = 'none';
                    document.getElementById('success-area').style.display = 'block';
                } else {
                    alert("Incorrect code. Try again.");
                }
            });
        }

        function verifyCaptcha() {
            const btn = document.getElementById('verify-btn');
            clearInterval(timerInterval);
            btn.innerText = "Verifying...";
            btn.disabled = true;
            
            fetch('/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, selected: selectedTiles })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('main-content').style.display = 'none';
                    document.getElementById('success-area').style.display = 'block';
                    
                    // Display the Ray ID
                    if(data.ray_id) {
                        document.getElementById('ray-id').innerText = data.ray_id;
                        document.getElementById('ray-id-container').style.display = 'block';
                    }
                } else {
                    btn.innerText = "Try again!";
                    btn.style.background = "red";
                    setTimeout(() => { window.location.reload(); }, 1000);
                }
            })
            .catch(err => {
                btn.innerText = "VERIFY";
                btn.disabled = false;
                alert("Server error. Please refresh.");
            });
        }
    </script>
</body>
</html>
"""

def center_crop_square(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top+min_dim, left:left+min_dim]

def load_image_safe(path, size):
    try:
        with Image.open(path) as img:
            img.draft('RGB', size)
            return img.convert("RGB").resize(size)
    except Exception:
        return None

def get_valid_targets(target_dict):
    valid = {}
    for k, v in target_dict.items():
        if os.path.exists(v['folder']):
            images = [f for f in os.listdir(v['folder']) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 0:
                valid[k] = v
    if not valid:
        valid[9] = YOLO_TARGETS[9]
    return valid

@app.route('/')
def index():
    stats["total_loads"] += 1
    session_id = str(uuid.uuid4())
    correct_indices = []

    challenge_level = session.get('challenge_level', 1)
    fail_count = session.get('fail_count', 0)
    show_checkbox = not session.get('skip_checkbox', False)
    
    if challenge_level == 1:
        grid_size = 3
        grid_class = "grid-3"
        available = get_valid_targets(YOLO_TARGETS)
        target_id = random.choice(list(available.keys()))
        target_info = available[target_id]
        challenge_target = target_info['name']
        target_icon = target_info['icon']
        folder_path = target_info['folder']
    else:
        grid_size = 4
        grid_class = "grid-4"
        available = get_valid_targets(MOBILENET_TARGETS)
        target_id = random.choice(list(available.keys()))
        target_info = available[target_id]
        challenge_target = target_info['name']
        target_icon = target_info['icon']
        folder_path = target_info['folder']
        
    total_tiles = grid_size * grid_size
    final_image = Image.new('RGB', (300, 300), color='#eeeeee')

    audio_code = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=5))
    captcha_data[session_id] = {'correct_indices': [], 'audio_code': audio_code, 'expires': time.time() + 120}

    if grid_size == 3:
        valid_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        attempts = 0
        chosen_image_path = None
        bad_images = set() 
        
        while attempts < 10:
            available_imgs = [img for img in valid_images if img not in bad_images]
            if not available_imgs: break
                
            chosen_image = random.choice(available_imgs)
            image_path = os.path.join(folder_path, chosen_image)
            img_cv = cv2.imread(image_path)
            
            if img_cv is None: 
                bad_images.add(chosen_image)
                attempts += 1
                continue
            
            img_cropped = center_crop_square(img_cv)
            img_resized = cv2.resize(img_cropped, (300, 300))
            blob = cv2.dnn.blobFromImage(img_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net_yolo.setInput(blob)
            outs = net_yolo.forward(output_layers)
            
            boxes, confidences = [], []
            temp_indices = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.35 and class_id == target_id: 
                        cx, cy, w, h = int(detection[0]*300), int(detection[1]*300), int(detection[2]*300), int(detection[3]*300)
                        x, y = max(0, cx - w//2), max(0, cy - h//2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.35, nms_threshold=0.4)
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        x_max, y_max = min(299, x+w), min(299, y+h)
                        for r in range(y//100, (y_max//100)+1):
                            for c in range(x//100, (x_max//100)+1):
                                idx = (r*3)+c
                                if idx not in temp_indices: temp_indices.append(idx)
            
            if 0 < len(temp_indices) <= 5: 
                correct_indices = temp_indices
                chosen_image_path = image_path
                break
            
            bad_images.add(chosen_image)
            attempts += 1
        
        if not correct_indices or chosen_image_path is None:
            correct_indices = [1, 4, 7]
            chosen_image_path = os.path.join(folder_path, random.choice(valid_images))

        with Image.open(chosen_image_path) as source_img:
            w, h = source_img.size
            min_dim = min(w, h)
            left, top = (w-min_dim)/2, (h-min_dim)/2
            source_img = source_img.crop((left, top, left+min_dim, top+min_dim)).resize((300, 300))
            
            for row in range(3):
                for col in range(3):
                    tile = source_img.crop((col*100, row*100, (col+1)*100, (row+1)*100))
                    final_image.paste(tile, (col*100, row*100))

    else:
        bg_folder = "real_images/backgrounds"
        other_folders = [d for d in os.listdir("real_images") if os.path.isdir(os.path.join("real_images", d)) and os.path.join("real_images", d) != folder_path]
        
        car_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(car_images)

        chosen_cars = []
        for c in car_images:
            if len(chosen_cars) == 4: break
            img = load_image_safe(os.path.join(folder_path, c), (75, 75))
            if img is not None: 
                chosen_cars.append(img)

        while len(chosen_cars) < 4:
            if chosen_cars:
                chosen_cars.append(random.choice(chosen_cars))
            else:
                chosen_cars.append(Image.new('RGB', (75, 75), color=(150, 150, 150)))

        bg_paths = []
        if os.path.exists(bg_folder):
            bg_paths.extend([os.path.join(bg_folder, f) for f in os.listdir(bg_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for other_f in other_folders:
            o_path = os.path.join("real_images", other_f)
            bg_paths.extend([os.path.join(o_path, f) for f in os.listdir(o_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        random.shuffle(bg_paths)

        chosen_bgs = []
        for p in bg_paths:
            if len(chosen_bgs) == 12: break
            img = load_image_safe(p, (75, 75))
            if img is not None: 
                chosen_bgs.append(img)
            
        while len(chosen_bgs) < 12:
            if chosen_bgs:
                chosen_bgs.append(random.choice(chosen_bgs))
            else:
                chosen_bgs.append(Image.new('RGB', (75, 75), color=(150, 150, 150)))

        car_indices_in_grid = random.sample(range(16), 4)
        correct_indices = car_indices_in_grid.copy()
        
        grid_items = [None] * 16
        for i, idx in enumerate(car_indices_in_grid):
            grid_items[idx] = chosen_cars[i]
            
        bg_idx = 0
        for i in range(16):
            if grid_items[i] is None:
                grid_items[i] = chosen_bgs[bg_idx]
                bg_idx += 1
                
        for idx, img in enumerate(grid_items):
            row = idx // 4
            col = idx % 4
            final_image.paste(img, (col*75, row*75))

    if grid_size == 3:
        angle = random.uniform(-12, 12)
    else:
        angle = random.uniform(-8, 8)
        
    final_image = final_image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor='#eeeeee')

    print(f"[Level {challenge_level} - {challenge_target}] Grid {grid_size}x{grid_size} | Indices: {correct_indices}")

    tiles = []
    tile_size = 300 // grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            left, top = col * tile_size, row * tile_size
            tile_img = final_image.crop((left, top, left + tile_size, top + tile_size))
            img_io = io.BytesIO()
            tile_img.save(img_io, 'PNG')
            img_io.seek(0)
            tiles.append(img_io.getvalue())

    captcha_data[session_id]['correct_indices'] = correct_indices
    captcha_data[session_id]['tiles'] = tiles
    
    show_skip = (fail_count >= 4 and challenge_level == 1)
    
    gc.collect()
    
    return render_template_string(HTML_PAGE, 
                                  session_id=session_id, 
                                  challenge_target=challenge_target, 
                                  target_icon=target_icon,
                                  grid_class=grid_class,
                                  total_tiles=total_tiles,
                                  show_skip=show_skip,
                                  show_checkbox=show_checkbox)

@app.route('/tile/<session_id>/<int:index>')
def get_tile(session_id, index):
    if session_id in captcha_data and 'tiles' in captcha_data[session_id] and index < len(captcha_data[session_id]['tiles']):
        return send_file(io.BytesIO(captcha_data[session_id]['tiles'][index]), mimetype='image/png')
    return "Not found", 404

@app.route('/audio/<session_id>')
def get_audio(session_id):
    if session_id in captcha_data:
        code = captcha_data[session_id]['audio_code']
        spoken_text = " ".join(list(code))
        try:
            tts = gTTS(text=spoken_text, lang='en', slow=True)
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            return send_file(audio_io, mimetype='audio/mpeg')
        except Exception as e:
            return "Audio generation failed", 500
    return "Not found", 404

@app.route('/verify_audio', methods=['POST'])
def verify_audio():
    data = request.json
    session_id = data.get('session_id')
    user_code = data.get('code', '').strip().upper()
    
    if session_id in captcha_data:
        if time.time() > captcha_data[session_id]['expires']:
            return jsonify({"success": False, "error": "Expired"})
            
        if user_code == captcha_data[session_id]['audio_code']:
            session['challenge_level'] = 1
            session['fail_count'] = 0
            session['skip_checkbox'] = False
            del captcha_data[session_id]
            return jsonify({"success": True})
            
    return jsonify({"success": False})

@app.route('/skip', methods=['POST'])
def skip():
    session['challenge_level'] = 2
    session['fail_count'] = 0
    return jsonify({"success": True})

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    session_id = data.get('session_id')
    selected = set(data.get('selected', []))
    
    if session_id in captcha_data:
        if time.time() > captcha_data[session_id]['expires']:
            return jsonify({"success": False, "error": "Expired"})
            
        correct = set(captcha_data[session_id]['correct_indices'])
        current_level = session.get('challenge_level', 1)
        
        if correct.issubset(selected) and (len(selected) - len(correct) <= 2):
            stats["success"] += 1
            if current_level == 1:
                stats["level1_success"] += 1
            else:
                stats["level2_success"] += 1
                
            # Generate Ray ID
            ray_id = uuid.uuid4().hex
            issued_tokens[ray_id] = time.time() + 300 # Valid for 5 minutes
            
            session['challenge_level'] = 1
            session['fail_count'] = 0 
            session['skip_checkbox'] = False
            del captcha_data[session_id]
            return jsonify({"success": True, "ray_id": ray_id})
        else:
            stats["failures"] += 1
            session['fail_count'] = session.get('fail_count', 0) + 1
            session['skip_checkbox'] = True
            return jsonify({"success": False})
            
    return jsonify({"success": False})

@app.route('/api/validate_ray_id', methods=['POST'])
def validate_ray_id():
    data = request.json
    ray_id = data.get('ray_id')
    
    if ray_id in issued_tokens:
        if time.time() < issued_tokens[ray_id]:
            return jsonify({"valid": True, "message": "Human verified"})
        else:
            del issued_tokens[ray_id]
            return jsonify({"valid": False, "error": "Ray ID expired"})
            
    return jsonify({"valid": False, "error": "Invalid Ray ID"})

@app.route('/admin')
def admin_dashboard():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CAPTCHA Admin Dashboard</title>
        <style>
            body {{ font-family: 'Roboto', sans-serif; background: #121212; color: white; padding: 40px; }}
            .card {{ background: #1e1e1e; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }}
            h1 {{ color: #1a73e8; }}
            h2 {{ color: #66b2ff; border-bottom: 1px solid #333; padding-bottom: 10px; }}
            .stat-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
            .stat-box {{ background: #2d2d2d; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-box h3 {{ margin: 0 0 10px 0; color: #aaa; font-size: 14px; text-transform: uppercase; }}
            .stat-box .value {{ font-size: 36px; font-weight: bold; color: #4CAF50; }}
            .stat-box .fail {{ color: #f44336; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🛡️ CAPTCHA Admin Dashboard</h1>
            <p>Real-time statistics for your AI security system.</p>
        </div>
        
        <div class="card">
            <h2>Overall Traffic</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <h3>Total Challenges Generated</h3>
                    <div class="value">{stats['total_loads']}</div>
                </div>
                <div class="stat-box">
                    <h3>Successful Verifications</h3>
                    <div class="value">{stats['success']}</div>
                </div>
                <div class="stat-box">
                    <h3>Failed Verifications</h3>
                    <div class="value fail">{stats['failures']}</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Challenge Difficulty Breakdown</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <h3>Level 1 (YOLO 3x3) Passes</h3>
                    <div class="value">{stats['level1_success']}</div>
                </div>
                <div class="stat-box">
                    <h3>Level 2 (MobileNet 4x4) Passes</h3>
                    <div class="value">{stats['level2_success']}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <p style="color: #888; font-size: 12px;">Note: These stats are stored in the server's memory. They will reset if you restart `python app.py`.</p>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
