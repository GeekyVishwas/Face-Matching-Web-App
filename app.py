import cv2
import numpy as np
import streamlit as st
import face_recognition
import base64

def image_to_base64(image):
    if image is None:
        return None
    _, buffer = cv2.imencode('.png', image)
    if buffer is None:
        return None
    return base64.b64encode(buffer).decode()

# Function to capture image from camera
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to open camera.")
        return None

    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Unable to capture frame from camera.")
        return None

    # Convert the frame to RGB (Streamlit uses RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Close the camera
    cap.release()

    return frame_rgb

# Function to match faces
def match_faces(image1, image2):
    image1_encoding = face_recognition.face_encodings(image1)
    image2_encoding = face_recognition.face_encodings(image2)

    if len(image1_encoding) == 0 or len(image2_encoding) == 0:
        return False, 0

    image1_encoding = image1_encoding[0]
    image2_encoding = image2_encoding[0]

    results = face_recognition.compare_faces([image1_encoding], image2_encoding)
    match_percentage = face_recognition.face_distance([image1_encoding], image2_encoding)

    return results[0], (1 - match_percentage[0]) * 100

# Streamlit app
def main():
    st.title("Face Matching Web App")

    session_state = st.session_state.setdefault('state', {
        'image1': None,
        'image2': None,
        'capture_first_image': False,
        'capture_second_image': False,
        'match_button_pressed': False,
    })

    if not session_state['capture_first_image'] and not session_state['capture_second_image']:
        session_state['image1'] = None
        session_state['image2'] = None

    uploaded_file1 = st.file_uploader("Upload first image", type=["png", "jpg", "jpeg"])
    uploaded_file2 = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])

    if not session_state['capture_first_image'] and not session_state['capture_second_image']:
        st.write("Please capture the first image:")
        session_state['capture_first_image'] = st.button("Capture First Image")

    if session_state['capture_first_image'] and session_state['image1'] is None:
        if uploaded_file1 is not None:
            session_state['image1'] = np.array(bytearray(uploaded_file1.read()), dtype=np.uint8)
            session_state['image1'] = cv2.imdecode(session_state['image1'], cv2.IMREAD_COLOR)
            if session_state['image1'] is not None:
                st.image(session_state['image1'], caption='Uploaded First Image', width=75, channels="BGR", use_column_width=False, clamp=True, output_format='png')
        else:
            session_state['image1'] = capture_image()
            if session_state['image1'] is not None:
                st.image(session_state['image1'], caption='Captured First Image', width=750, channels="BGR", use_column_width=False, clamp=True, output_format='png')

    if session_state['image1'] is not None and not session_state['capture_second_image']:
        st.write("Please capture the second image:")
        session_state['capture_second_image'] = st.button("Capture Second Image")

    if session_state['capture_second_image'] and session_state['image2'] is None:
        if uploaded_file2 is not None:
            session_state['image2'] = np.array(bytearray(uploaded_file2.read()), dtype=np.uint8)
            session_state['image2'] = cv2.imdecode(session_state['image2'], cv2.IMREAD_COLOR)
            if session_state['image2'] is not None:
                st.image(session_state['image2'], caption='Uploaded Second Image', width=75, channels="BGR", use_column_width=False, clamp=True, output_format='png')
        else:
            session_state['image2'] = capture_image()
            if session_state['image2'] is not None:
                st.image(session_state['image2'], caption='Captured Second Image', width=75, channels="BGR", use_column_width=False, clamp=True, output_format='png')

    if session_state['image1'] is not None and session_state['image2'] is not None:
        session_state['match_button_pressed'] = st.button("Match Faces")
        if session_state['match_button_pressed']:
            st.write("Matching...")

            match, match_percentage = match_faces(session_state['image1'], session_state['image2'])

            # Display both images side by side in smaller and rounded circles
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between;">
                    <img src="data:image/png;base64,{image_to_base64(session_state['image1'])}" alt="First Image" style="width:150px; border-radius:50%;">
                    <img src="data:image/png;base64,{image_to_base64(session_state['image2'])}" alt="Second Image" style="width:150px; border-radius:50%;">
                </div>
                """
                , unsafe_allow_html=True
            )

            if match:
                if match_percentage >= 60:
                    st.success(f"Faces match with {match_percentage:.2f}% accuracy")
                else:
                    st.error(f"Faces are not same with accuracy of {match_percentage:.2f}%")
            else:
                st.error("Faces do not match")

    st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between;">
                <img src="data:image/png;base64,{image_to_base64(session_state['image1'])}" alt="First Image" style="width:100px;">
                <img src="data:image/png;base64,{image_to_base64(session_state['image2'])}" alt="Second Image" style="width:100px;">
            </div>
            """
            , unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
