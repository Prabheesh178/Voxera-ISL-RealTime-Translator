"""
Voxera — webcam.py (Live Presentation Master Script)
Features: USB Phone Camera (DroidCam), Multi-Threaded TTS Voice, AI Stabilizer, NO MIRRORING
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pyttsx3
import threading

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────.
MODEL_PATH = r"D:\voxera_data\2_Model_Training\Model_2_Test\best_test_model.pth"
CONFIDENCE_THRESHOLD = 0.60  # AI must be 85% sure before it speaks
FRAMES_TO_CONFIRM = 10       # Must hold the sign for 10 frames (~0.3 seconds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Voxera AI on: {device}...")

# ─────────────────────────────────────────────────────────────
# 2. AUDIO ENGINE (Multi-Threaded to prevent camera lag)
# ─────────────────────────────────────────────────────────────
def speak_word(word):
    """Runs Text-to-Speech in a separate background thread."""
    def tts_task():
        # Initialize engine per-thread to prevent Windows COM errors
        engine = pyttsx3.init()
        engine.setProperty('rate', 160) # Speed of speech
        engine.say(word)
        engine.runAndWait()
        
    threading.Thread(target=tts_task, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# 3. LOAD MODEL & CLASSES DYNAMICALLY
# ─────────────────────────────────────────────────────────────
checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint["classes"]
num_classes = checkpoint["num_classes"]

print(f"Successfully loaded {num_classes} classes: {classes}")

# Rebuild the ResNet18 architecture exactly as it was trained
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)
)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval() # Set to evaluation mode!

# Image Transforms (Must match training exactly!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────
# 4. LIVE INFERENCE LOOP (USB Phone Camera Setup)
# ─────────────────────────────────────────────────────────────
print("\nConnecting to Motorola Phone Camera via DroidCam...")
cap = cv2.VideoCapture(1) 

if not cap.isOpened():
    print("\n[!] ERROR: Could not find the phone camera on Index 1.")
    print("[!] FIX: Change 'cv2.VideoCapture(1)' to a 2, 3, or 0 on line 65!")
    exit()

# Variables for smoothing output
current_sign = None
consecutive_frames = 0
last_spoken_sign = None

print("\nCamera Online! Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed lost. Exiting...")
        break

    # 🟢 FIX APPLIED: The mirror flip has been completely removed! 
    # The camera will now show exactly what the lens sees.

    # Prepare the image for the AI
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # AI Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_word = classes[predicted_idx.item()]
    conf_value = confidence.item()

    # ─────────────────────────────────────────────────────────────
    # 5. STABILIZER & VOICE LOGIC
    # ─────────────────────────────────────────────────────────────
    if conf_value >= CONFIDENCE_THRESHOLD:
        if predicted_word == current_sign:
            consecutive_frames += 1
        else:
            current_sign = predicted_word
            consecutive_frames = 1

        # If held long enough AND we haven't just spoken it
        if consecutive_frames >= FRAMES_TO_CONFIRM and current_sign != last_spoken_sign:
            speak_word(current_sign)
            last_spoken_sign = current_sign
            
        # UI: Draw Green Text for strong predictions
        display_text = f"{current_sign} ({conf_value*100:.1f}%)"
        text_color = (0, 255, 0) # Green
    else:
        # Reset if the hand drops or confidence is too low
        consecutive_frames = 0
        if conf_value < 0.40: # If the screen is mostly empty, allow the user to say the same word again
            last_spoken_sign = None 
            
        # UI: Draw Red Text for weak/uncertain predictions
        display_text = f"Uncertain: {predicted_word} ({conf_value*100:.1f}%)"
        text_color = (0, 0, 255) # Red

    # ─────────────────────────────────────────────────────────────
    # 6. ON-SCREEN UI OVERLAY
    # ─────────────────────────────────────────────────────────────
    # Background box for text readability
    cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1)
    
    cv2.putText(frame, display_text, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    cv2.imshow("Voxera Edge AI - Live Demo", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()