import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image
import mediapipe as mp
import tempfile


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic 
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation



DEMO_VIDEO = "Video.mp4"
DEMO_FACE = 'Face.jpg'
DEMO_BODY = 'Body.jpg'
AVATAR = "avata.jpg"
DEMO_HAND = 'Hand.jpg'





st.title('TEAM TN - HH | HACKATHON 2022 | COMPUTER VISION APPLICATIONS')
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Workspace - Options')
st.sidebar.subheader('Parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:

        r = width / float(w)
        dim = (width, int(h * r))

   
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['General information',
                                    'Face Detection', 'Body Detection', 'Real Time Testing', 'Hand Detection']
                                )

if app_mode == 'General information':
    st.markdown('Team TN-HH provides a website that is able to handle some computer vision problems. We believe these are fundamental tasks in order to tackle advanced problems related to Hacking AI Field. ')

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    
    image_profile = np.array(Image.open(AVATAR))
    scale_percent = 60 # percent of original size
    width_pro= int(image_profile.shape[1] * scale_percent / 100)
    height_pro = int(image_profile.shape[0] * scale_percent / 100)
    dim = (width_pro, height_pro)
    resized_pro = cv2.resize(image_profile, dim, interpolation = cv2.INTER_AREA)
    st.image(resized_pro)

    st.markdown('''
          # ABOUT US \n 
           Let's call this web **TN-HH tool**. There are a plenty of features that this web can operate.\n
           
            Hacking AI could be classified into two groups: **AI Against Hacking** and **AI To Hack**. \n
            
            Here are our fantastic features:
            - Face Detection
            - Body Detection 
            - Hand Detection
            - Real-team Detection

        
            It can be acknowledged that no matter what Types of hacking AI are, these are one of the most basic things we should approach if heading to this field. \n
            Since these are just ideas, it would be better if we have more time to continue with our work. Promisingly, the next version would surely be well-structured and effective.
             
            ''')
    
elif app_mode == 'Face Detection':

    drawing_spec = mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
    drawing_spec1 = mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=2)

    st.sidebar.markdown('---')

    st.markdown(
    """
    # Face Detection
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_FACE
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0
    
    
    # Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec1,
            connection_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('OUTPUT IMAGE')
        st.image(out_image,use_column_width= True)
 
        
elif app_mode == 'Real Time Testing':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        # Real Time Testing
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # max faces
    max_faces = st.sidebar.number_input(
        'Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider(
        'Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider(
        'Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## OUTPUT')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc(*'XVID')
    #codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.avi', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=max_faces) as face_mesh:
        prevTime = 0

        while vid.isOpened():
            i += 1
            ret, frame = vid.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                #st.checkbox("Recording", value=True)
                out.write(frame)
            #Dashboard
            kpi1_text.write(
                f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(
                f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(
                f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out. release()



elif app_mode =='Body Detection':
    
    drawing_spec = mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)
    drawing_spec1 = mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)
    
    
    st.sidebar.markdown('---')

    st.markdown(
    """
    # Body Detection
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    st.markdown('---')


    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])
    
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_BODY
        image = np.array(Image.open(demo_image))
    
    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0
    
    #mp_holistic
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:
        
        results = holistic.process(image)
        out_image = image.copy()
        

            
        mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=results.face_landmarks,
                connections=mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec1,
                connection_drawing_spec=drawing_spec)
        
        mp_drawing.draw_landmarks(
            out_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
        #kpi1_text.write(
               # f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        
        st.subheader('OUTPUT IMAGE')
        st.image(out_image,use_column_width= True)

elif app_mode == 'Hand Detection':
    
    
    st.sidebar.markdown('---')

    st.markdown(
        """
        # Hand Detection
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    

    st.markdown('---')

    max_hands = st.sidebar.number_input('Maximum Number of Hands', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider(
        'Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", 'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_HAND
        image = np.array(Image.open(demo_image))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence= detection_confidence) as hands:
        
        results = hands.process(image)
        out_image = image.copy()
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                out_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        st.subheader('OUTPUT IMAGE')
        st.image(out_image, use_column_width=True)

