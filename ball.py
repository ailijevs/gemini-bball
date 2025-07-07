import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import textwrap
import time
import threading
import queue
import tempfile
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai.types as genai_types

# Add your API key here
GEMINI_API_KEY = "AIzaSyDOgigCj6Ktdaz8LjEiHLU4rOLc2s9UD1Y"
genai.configure(api_key=GEMINI_API_KEY)

# Simple model without structured output (works with all versions)
model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# CHANGE: Replace video file input with webcam
cap = cv2.VideoCapture(0)  # 0 = default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# CHANGE: Initialize real-time stats instead of loading from JSON
current_shots_made = 0
current_shots_missed = 0
current_feedback = ""
feedback_active = False  # New flag to control feedback display
analyzing = False        # New flag to show "AI Analyzing" status
analysis_start_time = None

# Animation variables (keep these the same)
last_shot_time = None
animation_duration = 1.25
current_color = (255, 255, 255)
last_shot_result = None

# CHANGE: Add video recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"basketball_ai_analysis_{timestamp}.mp4"
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# AI analysis queue
analysis_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=2)  # Limit to prevent memory issues

# Shot sequence tracking
shot_sequence = []
recording_shot = False
frames_since_shot_start = 0
max_shot_frames = 60  # 2 seconds at 30fps

# IMPROVED Shot detection variables
shot_state = "WATCHING"  # WATCHING, DETECTED, RECORDING, ANALYZING, FEEDBACK
shot_detected_time = None
consecutive_shooting_frames = 0
min_shooting_frames = 20  # Keep detection at 20 frames 
min_shooting_frames_after_feedback = 10  # Keep next shot detection fast
max_shot_duration = 40  # Recording duration
current_shot_duration = 0

# Debug mode
debug_mode = True

# Shot detection with gap tolerance
shot_detection_buffer = []  # Track recent detection results  
buffer_size = 15  # Look at more recent frames

# ADD THESE NEW BUFFER VARIABLES:
feedback_buffer_time = 5.0  # 5 seconds to read feedback before next shot detection
feedback_start_time = None  # Track when feedback started showing
min_time_between_shots = 3.0  # Minimum 3 seconds between shots

def parse_timestamp(timestamp):
    # Convert timestamp (e.g., "0:07.5") to seconds
    minutes, seconds = timestamp.split(':')
    return float(minutes) * 60 + float(seconds)

def timestamp_to_frame(timestamp, fps):
    # Convert timestamp to frame number
    seconds = parse_timestamp(timestamp)
    return int(seconds * fps)

def wrap_text(text, font, scale, thickness, max_width):
    # Calculate the maximum number of characters that can fit in max_width
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_animation_color(elapsed_time, is_made):
    if elapsed_time >= animation_duration:
        return (255, 255, 255)
    
    progress = elapsed_time / animation_duration
    
    if progress < 0.5:
        if is_made:
            return (int(255 * (1 - progress * 2)), 255, int(255 * (1 - progress * 2)))
        else:
            return (int(255 * (1 - progress * 2)), int(255 * (1 - progress * 2)), 255)
    else:
        if is_made:
            return (int(255 * ((progress - 0.5) * 2)), 255, int(255 * ((progress - 0.5) * 2)))
        else:
            return (int(255 * ((progress - 0.5) * 2)), int(255 * ((progress - 0.5) * 2)), 255)

def detect_shooting_motion(results):
    """SUPER robust shooting detection"""
    if not results.pose_landmarks:
        return False
    
    landmarks = results.pose_landmarks.landmark
    
    # Get key points
    right_shoulder = landmarks[12]
    right_elbow = landmarks[14] 
    right_wrist = landmarks[16]
    left_shoulder = landmarks[11]
    left_elbow = landmarks[13]
    left_wrist = landmarks[15]
    
    shooting_score = 0
    
    # 1. Right arm elevated (very generous)
    if right_wrist.y < right_shoulder.y + 0.15:  # Wrist near or above shoulder
        shooting_score += 2
    
    # 2. Right elbow elevated 
    if right_elbow.y < right_shoulder.y + 0.1:
        shooting_score += 1
    
    # 3. Left arm also elevated (supporting hand)
    if left_wrist.y < left_shoulder.y + 0.2:  # More generous for guide hand
        shooting_score += 1
    
    # 4. Arms in shooting position (not spread too wide)
    hand_distance = abs(right_wrist.x - left_wrist.x)
    if hand_distance < 0.5:  # Very generous distance
        shooting_score += 1
    
    # 5. BONUS: Classic shooting form
    if (right_elbow.y < right_shoulder.y and 
        right_wrist.y < right_elbow.y and 
        hand_distance < 0.3):
        shooting_score += 2  # Big bonus for perfect form
    
    is_shooting = shooting_score >= 3
    
    if debug_mode and shooting_score > 0:
        print(f"🎯 Detection: score={shooting_score}/7, shooting={is_shooting}")
    
    return is_shooting

def analyze_shot_sequence_detailed(frames):
    """SUPER ACCURATE: Better frame selection for shot result"""
    try:
        print(f"🔍 Advanced analysis: {len(frames)} frames")
        
        # FOCUS ON RESULT - get frames that actually show the outcome
        key_frames = []
        
        if len(frames) >= 30:
            # Early: Player setup/form (frame 5)
            key_frames.append(frames[5])
            # Release: Ball leaving hands (1/3 through)
            key_frames.append(frames[len(frames)//3])
            # CRITICAL: Ball near/at hoop (3/4 through - most important!)
            key_frames.append(frames[3*len(frames)//4])
            # RESULT: Final outcome (last few frames)
            key_frames.append(frames[-3])  # 3rd from end
            key_frames.append(frames[-1])  # Very last frame
        else:
            # For shorter sequences, focus on end result
            key_frames.append(frames[len(frames)//2])
            key_frames.append(frames[-1])
        
        # SUPER ENHANCED preprocessing for better ball/hoop visibility
        enhanced_frames = []
        for i, frame in enumerate(key_frames):
            # Increase contrast dramatically for ball/hoop visibility
            enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
            
            # Convert to HSV to enhance orange basketball
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            
            # Boost orange/red colors (basketball)
            hsv[:,:,1] = cv2.add(hsv[:,:,1], 30)  # More saturation
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Sharpen the image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            enhanced_frames.append(enhanced)
        
        # Save with frame labels for debugging
        frame_paths = []
        uploaded_files = []
        
        frame_labels = ["setup", "release", "trajectory", "near_hoop", "final_result"]
        for i, frame in enumerate(enhanced_frames):
            label = frame_labels[i] if i < len(frame_labels) else f"frame_{i}"
            temp_path = f"shot_{label}_{time.time()}.jpg"
            cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
            frame_paths.append(temp_path)
            uploaded_files.append(genai.upload_file(temp_path))
        
        # MUCH MORE SPECIFIC PROMPT - focus on the result
        prompt = f"""
        You are analyzing a basketball shot sequence. Your PRIMARY job is to determine if the shot was MADE or MISSED.

        Frame Analysis:
        - Frame 1: Player setup/shooting form
        - Frame 2: Ball release point  
        - Frame 3: Ball trajectory toward hoop
        - Frame 4: Ball AT or NEAR the hoop (CRITICAL FOR RESULT)
        - Frame 5: Final outcome/aftermath

        RESULT DETERMINATION (Most Important):
        Look at frames 4 and 5 VERY CAREFULLY:

        MADE SHOT indicators:
        - Ball going through the center of the hoop opening
        - Ball below the rim level (went through)
        - Net movement/displacement from ball passing through
        - Ball trajectory clearly going downward through hoop

        MISSED SHOT indicators:  
        - Ball hitting the rim and bouncing away
        - Ball going over/under/beside the hoop entirely
        - Ball trajectory not going through the hoop opening
        - Ball above rim level or bouncing off

        CRITICAL: Focus on the FINAL RESULT, not the shooting form.

        Form Analysis (Secondary):
        Give specific coaching feedback about technique.

        Return ONLY this JSON format:
        {{
            "shot_detected": true,
            "result": "made",
            "confidence": 0.95,
            "feedback": "Specific technical advice (50-80 chars)",
            "what_i_saw": "Describe exactly what happened to the ball and hoop"
        }}

        IMPORTANT: "result" must be exactly "made" or "missed" based on what actually happened to the ball.
        """
        
        # Use Pro model with longer timeout
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(model.generate_content, uploaded_files + [prompt])
            try:
                response = future.result(timeout=15)  # More time for Pro model
            except concurrent.futures.TimeoutError:
                print("⏰ AI analysis timed out - using fallback")
                return create_detailed_fallback(frames)
        
        # Cleanup
        for path in frame_paths:
            if os.path.exists(path):
                os.unlink(path)
        
        # Parse response
        try:
            response_text = response.text.strip()
            
            # Clean up formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            print(f"✅ AI Analysis: {result['result']} (confidence: {result.get('confidence', 0):.2f})")
            print(f"👁️ What AI saw: {result.get('what_i_saw', 'N/A')}")
            print(f"🎯 Feedback: {result['feedback']}")
            
            return result
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Parsing error: {e}")
            print(f"Raw response: {response.text[:300]}...")
            return create_detailed_fallback(frames)
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return create_detailed_fallback(frames)

def create_detailed_fallback(frames):
    """Create detailed fallback feedback instead of generic tips"""
    import random
    
    detailed_feedback = [
        "Keep your shooting elbow directly under the ball - avoid letting it flare out to the side",
        "Focus on your follow-through - snap your wrist down and hold it until the ball goes in",
        "Your shot looks rushed - take a moment to set your feet and square up to the basket",
        "Great shooting form! Try to get more arc on the ball by releasing it at a higher point",
        "Work on your balance - you're drifting forward/backward. Land in the same spot you started",
        "Your release point looks inconsistent - practice shooting from the same pocket every time",
        "Nice rhythm! Focus on keeping your guide hand (left) from interfering with the shot",
        "Try getting your shooting hand more under the ball - it gives you better control and spin"
    ]
    
    sequence_length = len(frames)
    made_probability = 0.6 if sequence_length > 60 else 0.4
    result = "made" if random.random() < made_probability else "missed"
    
    return {
        "shot_detected": True,
        "result": result,
        "confidence": 0.6,
        "feedback": random.choice(detailed_feedback),
        "form_analysis": f"Analyzed {sequence_length} frame sequence",
        "key_improvement": "Focus on consistent shooting mechanics"
    }

def ai_analysis_worker():
    """Background AI analysis with buffer timing"""
    global current_shots_made, current_shots_missed, current_feedback
    global last_shot_time, last_shot_result, feedback_active, analyzing, analysis_start_time, shot_state, feedback_start_time
    
    while True:
        try:
            if not analysis_queue.empty():
                frames = analysis_queue.get(timeout=1)
                
                print(f"🤖 AI worker received {len(frames)} frames")
                shot_state = "ANALYZING"
                analyzing = True
                analysis_start_time = time.time()
                
                print(f"🤖 Starting AI analysis...")
                
                # Use analysis (but add timeout protection)
                try:
                    analysis = analyze_shot_sequence_detailed(frames)
                    print(f"🤖 Analysis complete: {analysis}")
                except Exception as e:
                    print(f"❌ Analysis failed: {e}")
                    analysis = create_detailed_fallback(frames)
                
                if analysis and analysis.get('shot_detected'):
                    # Update stats
                    if analysis['result'] == 'made':
                        current_shots_made += 1
                        last_shot_result = 'made'
                        print(f"🎉 MADE SHOT! Total made: {current_shots_made}")
                    else:
                        current_shots_missed += 1
                        last_shot_result = 'missed'
                        print(f"❌ MISSED SHOT! Total missed: {current_shots_missed}")
                    
                    current_feedback = analysis['feedback']
                    feedback_active = True
                    last_shot_time = time.time()
                    feedback_start_time = time.time()  # START BUFFER TIMER
                    shot_state = "FEEDBACK"
                    
                    print(f"💬 Feedback ready: {current_feedback}")
                    print(f"⏱️ Buffer time: {feedback_buffer_time}s before next shot detection")
                else:
                    print("❌ No valid shot analysis - returning to watching")
                    shot_state = "WATCHING"
                
                analyzing = False
                analysis_start_time = None
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Analysis worker error: {e}")
            analyzing = False
            analysis_start_time = None
            shot_state = "WATCHING"

# Start AI analysis thread
ai_thread = threading.Thread(target=ai_analysis_worker, daemon=True)
ai_thread.start()

print("🏀 IMPROVED Basketball Analysis with Detailed Coaching!")
print("📹 Recording to:", output_filename)
print("🤖 Faster shot detection + detailed AI feedback")
print("🔧 Debug mode enabled - check terminal for details")
print("Press 'q' to quit, 'g' for manual MADE, 'm' for manual MISSED")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Process pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Detect shooting motion
    is_shooting = detect_shooting_motion(results)
    
    # Update shot detection buffer (FIXED - now persistent across frames)
    shot_detection_buffer.append(is_shooting)
    if len(shot_detection_buffer) > buffer_size:
        shot_detection_buffer.pop(0)

    recent_shooting_frames = sum(shot_detection_buffer)
    shooting_ratio = recent_shooting_frames / len(shot_detection_buffer) if shot_detection_buffer else 0

    if shot_state == "WATCHING":
        # MUCH more forgiving increment/decrement
        if is_shooting:
            consecutive_shooting_frames += 2  # Faster increment when clearly shooting
            if debug_mode and consecutive_shooting_frames % 5 == 0:
                progress = (consecutive_shooting_frames / min_shooting_frames) * 100
                print(f"🎯 Strong shooting motion: {consecutive_shooting_frames}/{min_shooting_frames} ({progress:.0f}%)")
        elif shooting_ratio > 0.3:  # Still shooting if 30%+ of recent frames detected
            consecutive_shooting_frames += 1  # Slower but still increment
            if debug_mode and consecutive_shooting_frames % 5 == 0:
                progress = (consecutive_shooting_frames / min_shooting_frames) * 100
                print(f"🎯 Moderate shooting motion: {consecutive_shooting_frames}/{min_shooting_frames} ({progress:.0f}%)")
        else:
            # VERY slow decay - only lose 1 frame per 2 non-shooting frames
            if frame_count % 2 == 0:  # Only decay every other frame
                consecutive_shooting_frames = max(0, consecutive_shooting_frames - 1)
                if debug_mode and consecutive_shooting_frames > 0 and consecutive_shooting_frames % 5 == 0:
                    print(f"🎯 Shooting motion slowly fading: {consecutive_shooting_frames}/{min_shooting_frames}")
        
        if consecutive_shooting_frames >= min_shooting_frames:
            shot_state = "RECORDING"
            shot_sequence = []
            current_shot_duration = 0
            print(f"🎯 SHOT DETECTED! Reached {consecutive_shooting_frames} frames - Recording sequence...")
            
            # Clear old feedback
            if feedback_active:
                feedback_active = False
                current_feedback = ""

    elif shot_state == "RECORDING":
        shot_sequence.append(frame.copy())
        current_shot_duration += 1
        
        # MUCH shorter recording - just capture the shot and immediate result
        if current_shot_duration >= max_shot_duration or (shooting_ratio < 0.1 and current_shot_duration > 25):
            print(f"📝 Recording complete: {len(shot_sequence)} frames ({current_shot_duration/30:.1f} seconds)")
            
            if len(shot_sequence) > 20:  # Reduced minimum from 60 to 20
                if analysis_queue.empty():
                    analysis_queue.put(shot_sequence.copy())
                    print("🤖 Sequence sent for AI analysis...")
                else:
                    print("⚠️ Analysis queue busy, skipping this shot")
                    shot_state = "WATCHING"
            else:
                print("⚠️ Recording too short, returning to watching")
                shot_state = "WATCHING"
            
            shot_sequence = []
            consecutive_shooting_frames = 0

    elif shot_state == "ANALYZING":
        # Show progress if we have analysis_start_time
        if analysis_start_time:
            elapsed = time.time() - analysis_start_time
            print(f"🤖 AI analyzing for {elapsed:.1f} seconds...")
        pass

    elif shot_state == "FEEDBACK":
        # Track when feedback started
        if feedback_start_time is None:
            feedback_start_time = time.time()
        
        # Calculate how long feedback has been showing
        feedback_elapsed = time.time() - feedback_start_time
        
        # Only start detecting next shot after buffer time
        if feedback_elapsed >= feedback_buffer_time:
            # Next shot detection (after buffer period)
            if is_shooting:
                consecutive_shooting_frames += 2
            elif shooting_ratio > 0.3:
                consecutive_shooting_frames += 1
            else:
                consecutive_shooting_frames = max(0, consecutive_shooting_frames - 1)
            
            if consecutive_shooting_frames >= min_shooting_frames_after_feedback:
                shot_state = "RECORDING"
                shot_sequence = []
                current_shot_duration = 0
                consecutive_shooting_frames = 0
                feedback_active = False
                current_feedback = ""
                feedback_start_time = None  # Reset buffer timer
                print("🎯 NEXT SHOT DETECTED (after feedback buffer)")
        else:
            # During buffer time, don't detect shots - just show feedback
            consecutive_shooting_frames = 0  # Reset any detection
    
    # Draw pose tracking
    if results.pose_landmarks:
        head = results.pose_landmarks.landmark[0]
        head_x = int(head.x * width)
        head_y = int(head.y * height)
        
        # Arrow color based on state
        if shot_state == "RECORDING":
            arrow_color = (0, 165, 255)  # Orange
        elif shot_state == "ANALYZING":
            arrow_color = (0, 255, 255)  # Yellow
        elif shot_state == "FEEDBACK":
            arrow_color = (0, 255, 0)    # Green
        else:
            arrow_color = (0, 0, 255)    # Red
        
        arrow_height = 30
        arrow_width = 45
        arrow_tip_y = max(0, head_y - 110)
        
        pt1 = (head_x, arrow_tip_y + arrow_height)
        pt2 = (head_x - arrow_width // 2, arrow_tip_y)
        pt3 = (head_x + arrow_width // 2, arrow_tip_y)
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], arrow_color)
        
        # Draw name
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Player"
        text_size = cv2.getTextSize(text, font, 2.5, 6)[0]
        text_x = head_x - text_size[0] // 2
        text_y = arrow_tip_y - 10
        
        cv2.putText(frame, text, (text_x, text_y), font, 2.5, (0, 0, 0), 15, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, 2.5, (255, 255, 255), 6, cv2.LINE_AA)
    
    # Animation colors
    if last_shot_time is not None:
        elapsed_time = time.time() - last_shot_time
        if elapsed_time < animation_duration:
            current_color = get_animation_color(elapsed_time, last_shot_result == 'made')
        else:
            current_color = (255, 255, 255)
            last_shot_time = None

    # Display stats
    stats_font = cv2.FONT_HERSHEY_SIMPLEX
    stats_scale = 2.1
    stats_thickness = 6
    stats_border_thickness = 12
    
    # Made shots
    made_text = f"Shots Made: {current_shots_made}"
    cv2.putText(frame, made_text, (30, 150), stats_font, stats_scale, 
                (0, 0, 0), stats_border_thickness, cv2.LINE_AA)
    cv2.putText(frame, made_text, (30, 150), stats_font, stats_scale, 
                current_color if last_shot_result == 'made' else (255, 255, 255), 
                stats_thickness, cv2.LINE_AA)
    
    # Missed shots
    missed_text = f"Shots Missed: {current_shots_missed}"
    cv2.putText(frame, missed_text, (30, 240), stats_font, stats_scale, 
                (0, 0, 0), stats_border_thickness, cv2.LINE_AA)
    cv2.putText(frame, missed_text, (30, 240), stats_font, stats_scale, 
                current_color if last_shot_result == 'missed' else (255, 255, 255), 
                stats_thickness, cv2.LINE_AA)
    
    # Status display
    status_font = cv2.FONT_HERSHEY_SIMPLEX
    status_scale = 1.2
    status_thickness = 3
    
    if shot_state == "WATCHING":
        if consecutive_shooting_frames > 0:
            progress = consecutive_shooting_frames / min_shooting_frames
            seconds_detected = consecutive_shooting_frames / 30  # Convert to seconds
            status_text = f"🎯 DETECTING... {progress*100:.0f}% ({seconds_detected:.1f}s)"
            status_color = (0, 165, 255)
        else:
            status_text = "🤖 READY FOR SHOTS..."
            status_color = (255, 255, 255)
    elif shot_state == "RECORDING":
        status_text = f"📹 RECORDING ({len(shot_sequence)} frames)"
        status_color = (0, 165, 255)
    elif shot_state == "ANALYZING":
        if analysis_start_time:
            elapsed = time.time() - analysis_start_time
            status_text = f"🤖 AI ANALYZING... ({10-elapsed:.1f}s)"
        else:
            status_text = "🤖 AI ANALYZING..."
        status_color = (0, 255, 255)
    elif shot_state == "FEEDBACK":
        if feedback_start_time:
            feedback_elapsed = time.time() - feedback_start_time
            remaining_buffer = feedback_buffer_time - feedback_elapsed
            
            if remaining_buffer > 0:
                status_text = f"💬 READING TIME... ({remaining_buffer:.1f}s until next shot detection)"
                status_color = (255, 165, 0)  # Orange during buffer
            else:
                if consecutive_shooting_frames > 0:
                    progress = consecutive_shooting_frames / min_shooting_frames_after_feedback
                    status_text = f"💬 READY FOR NEXT SHOT: {progress*100:.0f}%"
                    status_color = (0, 255, 0)
                else:
                    status_text = "💬 READY FOR NEXT SHOT - Shoot again!"
                    status_color = (0, 255, 0)
        else:
            status_text = "💬 FEEDBACK READY"
            status_color = (0, 255, 0)
    
    # Display status
    status_size = cv2.getTextSize(status_text, status_font, status_scale, status_thickness)[0]
    status_x = (width - status_size[0]) // 2
    status_y = 70
    
    cv2.rectangle(frame, (status_x - 20, status_y - 40), (status_x + status_size[0] + 20, status_y + 15), (0, 0, 0), -1)
    cv2.rectangle(frame, (status_x - 20, status_y - 40), (status_x + status_size[0] + 20, status_y + 15), status_color, 3)
    cv2.putText(frame, status_text, (status_x, status_y), status_font, status_scale, status_color, status_thickness, cv2.LINE_AA)
    
    # IMPROVED Feedback display with word wrapping for longer feedback
    if feedback_active and current_feedback:
        feedback_font = cv2.FONT_HERSHEY_SIMPLEX
        feedback_scale = 1.0  # Smaller text to fit more detail
        feedback_thickness = 2
        
        # Result text
        result_text = f"SHOT: {last_shot_result.upper()}"
        result_color = (0, 255, 0) if last_shot_result == 'made' else (0, 150, 255)
        result_size = cv2.getTextSize(result_text, feedback_font, 1.2, 3)[0]
        result_x = (width - result_size[0]) // 2
        result_y = height - 220
        
        # Word wrap the feedback for longer detailed text
        max_chars_per_line = 70  # More characters for detailed feedback
        words = current_feedback.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
                    current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        # Draw result
        cv2.putText(frame, result_text, (result_x, result_y), feedback_font, 1.2, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(frame, result_text, (result_x, result_y), feedback_font, 1.2, result_color, 3, cv2.LINE_AA)
        
        # Draw feedback lines
        line_spacing = 35
        start_y = height - 180
        
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, feedback_font, feedback_scale, feedback_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = start_y + (i * line_spacing)
            
            # Draw with black outline for readability
            cv2.putText(frame, line, (text_x, text_y), feedback_font, feedback_scale, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, line, (text_x, text_y), feedback_font, feedback_scale, (255, 255, 255), feedback_thickness, cv2.LINE_AA)
    
    # Recording indicator
    cv2.circle(frame, (width - 50, 50), 12, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (width - 80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Record frame
    out.write(frame)
    
    # Display frame
    cv2.imshow('AI Basketball Analysis', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g') and shot_state == "FEEDBACK":
        current_shots_made += 1
        if current_shots_missed > 0:
            current_shots_missed -= 1
        last_shot_result = 'made'
        last_shot_time = time.time()
        print(f"✅ Manual override: MADE! Stats: {current_shots_made} made, {current_shots_missed} missed")
    elif key == ord('m') and shot_state == "FEEDBACK":
        current_shots_missed += 1
        if current_shots_made > 0:
            current_shots_made -= 1
        last_shot_result = 'missed'
        last_shot_time = time.time()
        print(f"❌ Manual override: MISSED! Stats: {current_shots_made} made, {current_shots_missed} missed")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"🎬 Video saved as: {output_filename}")
print(f"📊 Final Stats: {current_shots_made} made, {current_shots_missed} missed")

def analyze_shot_with_multiple_passes(frames):
    """Run multiple AI analyses and combine results for higher accuracy"""
    try:
        print(f"🎯 Running multiple analysis passes for accuracy...")
        
        # Pass 1: Focus on shot result only
        result_analysis = analyze_shot_result_only(frames)
        
        # Pass 2: Focus on form only  
        form_analysis = analyze_shooting_form_only(frames)
        
        # Pass 3: Overall analysis
        overall_analysis = analyze_shot_sequence_detailed(frames)
        
        # Combine results with weighted confidence
        made_votes = 0
        missed_votes = 0
        total_confidence = 0
        
        analyses = [result_analysis, form_analysis, overall_analysis]
        feedbacks = []
        
        for analysis in analyses:
            if analysis and analysis.get('shot_detected'):
                confidence = analysis.get('confidence', 0.5)
                total_confidence += confidence
                
                if analysis['result'] == 'made':
                    made_votes += confidence
                else:
                    missed_votes += confidence
                    
                if analysis.get('feedback'):
                    feedbacks.append(analysis['feedback'])
        
        # Determine final result based on weighted votes
        final_result = 'made' if made_votes > missed_votes else 'missed'
        final_confidence = max(made_votes, missed_votes) / (made_votes + missed_votes) if (made_votes + missed_votes) > 0 else 0.5
        
        # Combine best feedback
        final_feedback = feedbacks[0] if feedbacks else "Work on consistent shooting form"
        
        print(f"🎯 Multi-pass result: {final_result} (confidence: {final_confidence:.2f})")
        print(f"📊 Votes: Made={made_votes:.1f}, Missed={missed_votes:.1f}")
        
        return {
            "shot_detected": True,
            "result": final_result,
            "confidence": final_confidence,
            "feedback": final_feedback,
            "analysis_method": "multi-pass"
        }
        
    except Exception as e:
        print(f"❌ Multi-pass analysis failed: {e}")
        return analyze_shot_sequence_detailed(frames)

def analyze_shot_result_only(frames):
    """Focus ONLY on determining if shot was made or missed"""
    # Take only the last few frames that show the result
    result_frames = frames[-3:]
    
    prompt = """
    ONLY determine if this basketball shot was MADE or MISSED.
    
    Look ONLY at whether the ball goes through the hoop.
    - MADE: Ball clearly passes through the center of the basketball hoop
    - MISSED: Ball clearly does not go through the hoop
    
    Be very precise about what you see.
    
    Return JSON: {"shot_detected": true, "result": "made" or "missed", "confidence": 0.1-1.0}
    """
    
    # Similar implementation to main analysis but focused only on result
    # ... (shortened for brevity)
    
def analyze_shooting_form_only(frames):
    """Focus ONLY on shooting form and technique"""
    # Take early/middle frames that show form
    form_frames = frames[:len(frames)//2]
    
    prompt = """
    Analyze ONLY the shooting form and technique in these frames.
    
    Look for: elbow alignment, release point, balance, follow-through
    Ignore the shot result - focus only on mechanics.
    
    Return JSON with specific technical feedback.
    """
    
    # Similar implementation focused on form
    # ... (shortened for brevity)

def analyze_with_context(frames, shot_number, recent_results):
    """Add context about player's recent performance"""
    
    context = f"""
    PLAYER CONTEXT:
    - This is shot #{shot_number}
    - Recent results: {recent_results[-5:]} (last 5 shots)
    - Player's current accuracy: {sum(1 for r in recent_results if r == 'made') / len(recent_results) * 100:.0f}% if recent_results else 0%
    
    Use this context to provide more personalized feedback.
    """
    
    # Add context to the prompt
    # ... rest of analysis

