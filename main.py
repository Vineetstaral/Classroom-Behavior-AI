import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import pandas as pd

# Classroom-specific configuration
CLASSROOM_BEHAVIORS = {
    "engaged": {"color": (0, 200, 0), "emoji": "✅", "triggers": ["happy", "surprise"]},
    "confused": {"color": (255, 165, 0), "emoji": "🤔", "triggers": ["sad", "fear"]},
    "distracted": {"color": (255, 255, 100), "emoji": "😑", "triggers": ["neutral"]},
    "frustrated": {"color": (255, 50, 50), "emoji": "😠", "triggers": ["angry", "disgust"]}
}

# Initialize face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_classroom_image(img):
    """Process uploaded image for classroom behaviors"""
    # Convert to OpenCV format
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(50, 50))
    
    results = []
    behavior_counts = {behavior: 0 for behavior in CLASSROOM_BEHAVIORS}
    behavior_counts["unknown"] = 0
    
    for (x, y, w, h) in faces:
        face_img = img_bgr[y:y+h, x:x+w]
        
        try:
            # Analyze emotion (skip if takes more than 1 second per face)
            analysis = DeepFace.analyze(face_img, actions=['emotion'], 
                                      detector_backend='opencv', silent=True, timeout=1)
            
            # Map to classroom behavior
            emotion = max(analysis[0]['emotion'].items(), key=lambda x: x[1])[0]
            detected_behavior = "unknown"
            
            for behavior, config in CLASSROOM_BEHAVIORS.items():
                if emotion in config["triggers"]:
                    detected_behavior = behavior
                    break
            
            behavior_counts[detected_behavior] += 1
            
            # Store results for visualization
            results.append({
                "box": (x, y, w, h),
                "behavior": detected_behavior,
                "color": CLASSROOM_BEHAVIORS.get(detected_behavior, {}).get("color", (200, 200, 200))
            })
            
        except Exception as e:
            continue
    
    return img_array, results, behavior_counts

# Streamlit Interface
st.title("📚 Classroom Behavior Analyzer")
st.markdown("Upload a classroom photo to analyze student engagement and emotions")

uploaded_file = st.file_uploader("Choose classroom image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and process image
    img = Image.open(uploaded_file)
    
    with st.spinner("Analyzing student behaviors..."):
        processed_img, faces, counts = analyze_classroom_image(img)
    
    # Visualization
    st.subheader("Classroom Snapshot")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Draw annotations on image
        img_display = processed_img.copy()
        for face in faces:
            x, y, w, h = face["box"]
            cv2.rectangle(img_display, (x, y), (x+w, y+h), face["color"], 3)
            cv2.putText(img_display, 
                       face["behavior"].upper(), 
                       (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       face["color"], 
                       2)
        
        st.image(img_display, use_column_width=True)
    
    with col2:
        # Display statistics
        st.subheader("Behavior Summary")
        
        # Create summary table
        summary_data = []
        for behavior, config in CLASSROOM_BEHAVIORS.items():
            summary_data.append({
                "Behavior": f"{config['emoji']} {behavior.title()}",
                "Count": counts[behavior],
                "Percentage": f"{counts[behavior]/sum(counts.values())*100:.1f}%"
            })
        
        st.table(pd.DataFrame(summary_data))
        
        # Interpretation guide
        st.markdown("""
        **How to Interpret:**
        - ✅ Engaged: Actively participating
        - 🤔 Confused: May need clarification
        - 😑 Distracted: Possibly disengaged
        - 😠 Frustrated: Might need assistance
        """)

    # Detailed findings section
    st.subheader("Detailed Findings")
    
    if sum(counts.values()) > 0:
        # Engagement metric
        engagement_score = (counts["engaged"] / sum(counts.values())) * 100
        st.metric("Overall Engagement Score", f"{engagement_score:.1f}%")
        
        # Recommendations
        if counts["confused"] > counts["engaged"]:
            st.warning("⚠️ Many confused students - consider reviewing this lesson segment")
        elif counts["distracted"] > sum(counts.values())/2:
            st.warning("⚠️ High distraction level - try more interactive activities")
        else:
            st.success("✓ Good overall engagement")
    else:
        st.warning("No students detected in the image")

# Add teacher tips in sidebar
st.sidebar.markdown("""
### Teacher's Guide
**Best Practices:**
1. Capture from the front at student eye-level
2. Ensure good lighting (avoid backlight)
3. Take photos during active lessons

**When to Use:**
- During lectures
- Group work sessions
- Q&A periods

**Limitations:**
- Works best with clear frontal faces
- May miss subtle expressions
- Not a replacement for human observation
""")
