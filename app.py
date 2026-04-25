import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
from collections import deque
import time
import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Emotion Analytics", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
h1 {
    text-align: center;
    color: #38bdf8;
}
.block-container {
    padding-top: 1rem;
}
.stButton>button {
    border-radius: 12px;
    padding: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🧠 AI Emotion Analytics Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

if st.sidebar.button("▶ Start Camera"):
    st.session_state.run = True

if st.sidebar.button("⏹ Stop Camera"):
    st.session_state.run = False

st.sidebar.markdown("---")

if st.sidebar.button("💾 Save Data"):
    if st.session_state.emotion_history:
        df = pd.DataFrame(st.session_state.emotion_history)
        df.to_csv("emotion_log.csv", index=False)
        st.sidebar.success("Saved as emotion_log.csv")
    else:
        st.sidebar.warning("No data to save!")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])

FRAME_WINDOW = col1.image([])
chart = col2.empty()

col3, col4 = st.columns(2)
history_chart = col3.empty()
recommendation_box = col4.empty()
fitness_box = col4.empty()

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

emotion_buffer = deque(maxlen=5)

# ---------------- HELPERS ----------------
def get_recommendation(emotion):
    return {
        "happy": "😊 Keep smiling! Spread positivity.",
        "sad": "😢 Take a break, listen to music.",
        "angry": "😠 Try deep breathing.",
        "fear": "😨 Stay calm, you are safe.",
        "surprise": "😲 Stay curious!",
        "neutral": "😐 Try something new!",
        "disgust": "🤢 Refresh your mind."
    }.get(emotion, "")

def fitness_recommendation(emotion):
    return {
        "happy": "🔥 Maintain your workout routine!",
        "sad": "🧘 Try yoga or stretching",
        "angry": "🏃 High-intensity workout",
        "fear": "🚶 Go for a walk",
        "neutral": "💪 Light exercise",
        "surprise": "🤸 Try dance",
        "disgust": "🌿 Relax outside"
    }.get(emotion, "")

# ---------------- CAMERA ----------------
camera = cv2.VideoCapture(0)
frame_count = 0

while st.session_state.run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not working")
        break

    frame_count += 1

    # Skip frames for performance
    if frame_count % 3 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            emotions = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']

            # Smooth prediction
            emotion_buffer.append(dominant_emotion)
            stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

            # Store with timestamp
            st.session_state.emotion_history.append({
                "time": datetime.datetime.now(),
                "emotion": stable_emotion
            })

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Label
            cv2.putText(frame, stable_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Live chart
            df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Confidence'])
            chart.bar_chart(df.set_index('Emotion'))

            # Recommendation cards
            recommendation_box.info(get_recommendation(stable_emotion))
            fitness_box.success(fitness_recommendation(stable_emotion))

    except Exception as e:
        st.warning(f"Error: {e}")

    # Show frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # ---------------- TIMELINE GRAPH (IMPROVED) ----------------
    if len(st.session_state.emotion_history) > 5:
        history_df = pd.DataFrame(st.session_state.emotion_history)

        # Convert time to string for better plotting
        history_df["time"] = history_df["time"].dt.strftime("%H:%M:%S")

        timeline_df = history_df.groupby(["time", "emotion"]).size().unstack().fillna(0)

        history_chart.line_chart(timeline_df)

    time.sleep(0.03)

camera.release()