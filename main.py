import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import pandas as pd

# Classroom configuration with better face detection parameters
CLASSROOM_BEHAVIORS = {
    "engaged": {"color": (0, 200, 0), "triggers": ["happy", "surprise"]},
    "confused": {"color": (255, 165, 0),  "triggers": ["sad", "fear"]},
    "distracted": {"color": (255, 255, 100),  "triggers": ["neutral"]},
    "frustrated": {"color": (255, 50, 50), "triggers": ["angry", "disgust"]}
}

# Enhanced face detection parameters
def initialize_detector():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def analyze_classroom_image(img):
    """Improved face detection and analysis"""
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Enhanced detection parameters for classroom settings
    face_cascade = initialize_detector()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    behavior_counts = {behavior: 0 for behavior in CLASSROOM_BEHAVIORS}
    behavior_counts["unknown"] = 0
    
    for (x, y, w, h) in faces:
        face_img = img_bgr[y:y+h, x:x+w]
        
        try:
            analysis = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            
            emotion = max(analysis[0]['emotion'].items(), key=lambda x: x[1])[0]
            detected_behavior = next(
                (b for b, config in CLASSROOM_BEHAVIORS.items() 
                 if emotion in config["triggers"]),
                "unknown"
            )
            
            behavior_counts[detected_behavior] += 1
            results.append({
                "box": (x, y, w, h),
                "behavior": detected_behavior,
                "color": CLASSROOM_BEHAVIORS.get(detected_behavior, {}).get("color", (200, 200, 200))
            })
            
        except Exception as e:
            continue
    
    return img_array, results, behavior_counts

# Streamlit Interface
st.set_page_config(layout="wide")
st.title("Smart Classroom Analyzer")

uploaded_file = st.file_uploader("Upload classroom photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    with st.spinner("Detecting and analyzing faces..."):
        processed_img, faces, counts = analyze_classroom_image(img)
    
    # Visualization
    st.subheader("Classroom Analysis")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if faces:
            img_display = processed_img.copy()
            for face in faces:
                x, y, w, h = face["box"]
                cv2.rectangle(img_display, (x, y), (x+w, y+h), face["color"], 3)
                cv2.putText(
                    img_display,
                    f"{face['behavior'].upper()}",
                    (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    face["color"],
                    2
                )
            st.image(img_display, use_container_width=True)
        else:
            st.warning("⚠️ No faces detected. Try:")
            st.markdown("""
            - Clearer/higher resolution image
            - Front-facing student photos
            - Better lighting conditions
            """)
            st.image(processed_img, use_container_width=True)
    
    with col2:
        total_students = sum(counts.values())
        if total_students > 0:
            st.success(f"Detected {total_students} student(s)")
            
            # Behavior summary
            summary_data = []
            for behavior, config in CLASSROOM_BEHAVIORS.items():
                percentage = (counts[behavior] / total_students) * 100
                summary_data.append({
                    "Behavior": f"{behavior.title()}",
                    "Count": counts[behavior],
                    "Percent": f"{percentage:.1f}%"
                })
            
            st.dataframe(
                pd.DataFrame(summary_data),
                column_config={
                    "Behavior": "Behavior",
                    "Count": st.column_config.NumberColumn("Count"),
                    "Percent": st.column_config.ProgressColumn(
                        "Percent",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Engagement metrics
            engagement = (counts["engaged"] / total_students) * 100
            st.metric("Engagement Score", f"{engagement:.1f}%")
            
            # Suggestions
            if counts["confused"] > counts["engaged"]:
                st.warning("Consider reviewing the current lesson segment")
            elif counts["distracted"] > total_students/2:
                st.warning("Try more interactive teaching methods")
        else:
            st.warning("No analyzable faces found")
