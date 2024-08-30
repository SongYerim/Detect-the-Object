import streamlit as st
import os
import yaml
import shutil
from pathlib import Path
import base64

import streamlit as st

# 사이드바에 제목 추가
st.sidebar.title("Help")

def sidebar_image(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    encoded_img = base64.b64encode(img_data).decode()
    img_html = f'<img src="data:image/png;base64,{encoded_img}" style="width:100%;">'
    st.sidebar.markdown(img_html, unsafe_allow_html=True)
    
st.sidebar.markdown("""
### Tutorial 
**[Training the model]**
1. **Before training the model**, create a dataset by following the instructions in the Before training the Model link.
2. **train_data** should consist of two folders: `images` and `labels`. 
   Each folder consists of two subfolders: `train` and `val`.
   Upload each folder by dragging or dropping it to the appropriate location.
""")
sidebar_image("../yolov5/folders.png")
st.sidebar.markdown("""
3. **Validation** should be performed on a separate validation set.
    (Usually about 5% of the dataset is used for validation.)
4. Enter the number of labels used when creating the dataset in the **Enter number of classes (ex: 5)** field.
5. Enter the names of the labels you specified when creating the dataset in the **Enter class names (comma separated, ex: L1, L2, L3, L4, L5)** field.
6. If you performed steps 3 and 4, click the **Save Configuration** button.
If the phrase “Configuration saved to corning.yaml” appears below the button, it is successful.
7. Select the number of times to train the model repeatedly in the **Enter number of epochs** field.
8. The **Train the Model** button, which allows you to train the model, is activated only after step 5 is performed.
""")
st.sidebar.markdown("""
**[Detect Image]**
1. To evaluate the performance of your model, you need to perform tests (corresponding to steps 1-7) on the test set.
2. Upload the image you want to test by dragging or dropping it to the appropriate location.
3. Click the Run YOLO Model button to run image detection.
4. **Detection** refers to the label name of the detected image.
 **Coordinates** refer to the positions of the four corners of the square that appear in the detected image.
 **Accuracy** refers to prediction accuracy.
""")

# Function to set background image
def set_background(image_path):
    
    with open(image_path, "rb") as f:
        data = f.read()
    encoded_image = base64.b64encode(data).decode()
    background_image = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .title-container {{
        margin-top: 20px;
        margin-bottom: 10px;
    }}
    .title-container h1 {{
        font-size: 80px;  /* 글씨 크기 키우기 */
        color: white;  /* 글씨 색상 변경 */
    }}
    .column-container {{
        display: flex;
        justify-content: space-between;
        gap: 80px;
    }}
    .sidebar .sidebar-content {{
        background-color: #E9D8F6;
    }}
    .file-upload-container {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
    }}
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

# Function to display HTML file content
def display_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    st.markdown(html_content, unsafe_allow_html=True)

# Path to the background image
background_image_path = "../yolov5/background_design.png"

# Set the background image
set_background(background_image_path)

# Add CSS classes to the title and columns
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.title("Detect the Object !")
st.markdown('</div>', unsafe_allow_html=True)

# Add vertical space between title and content
st.markdown("<br>", unsafe_allow_html=True)

# Create two columns with a gap
col1, col2, col3 = st.columns([4.8, 0.2, 4.8])

with col1:
    st.header("Training the model")

    # External link to the labeling tool
    st.markdown("[Before training the Model](https://hospitable-arithmetic-3f5.notion.site/Before-training-the-Model-c09b1e13ae69450e8490941168bf6978?pvs=4)")

    # Upload train images and val images
    st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
    train_images = st.file_uploader("Upload train images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    val_images = st.file_uploader("Upload validation images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload train labels and val labels
    st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
    train_labels = st.file_uploader("Upload train labels", accept_multiple_files=True, type=['txt'])
    val_labels = st.file_uploader("Upload validation labels", accept_multiple_files=True, type=['txt'])
    st.markdown('</div>', unsafe_allow_html=True)

    # Number of classes input
    nc = st.number_input("Enter number of classes (ex: 5)", min_value=1) 

    # Class names input with placeholder guidance in the label
    class_names = st.text_area("Enter class names (comma separated, ex: L1, L2, L3, L4, L5)", value="")

    # Initialize session state for configuration saved
    if "config_saved" not in st.session_state:
        st.session_state.config_saved = False

    # Save the .yaml configuration file
    def save_yaml(nc, class_names):
        data = {
            'path': '../train_data',
            'train': 'images/train',
            'val': 'images/val',
            'test': '',
            'nc': nc,
            'names': class_names.split(',')
        }
        with open("/home/team04/yolov5/data/corning.yaml", 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

    # Save configuration button
    if st.button('Save Configuration'):
        save_yaml(nc, class_names)
        st.success("Configuration saved to corning.yaml")
        st.session_state.config_saved = True

    # Epochs input
    epochs = st.number_input("Enter number of epochs", min_value=1, value=3)

    # Training button
    if not st.session_state.config_saved:
        st.warning("Please save the configuration file before training the model.")
    else:
        if st.button('Train YOLO Model'):
            if train_images and train_labels and val_images and val_labels:
                base_dir = "/home/team04/train_data/"
                images_dir = os.path.join(base_dir, "images")
                labels_dir = os.path.join(base_dir, "labels")

                # 기존 train_data 폴더의 내용 삭제
                if os.path.exists(images_dir):
                    shutil.rmtree(images_dir)
                if os.path.exists(labels_dir):
                    shutil.rmtree(labels_dir)

                # 새로운 train_data 폴더 생성
                for dir_path in [images_dir, labels_dir]:
                    os.makedirs(os.path.join(dir_path, "train"), exist_ok=True)
                    os.makedirs(os.path.join(dir_path, "val"), exist_ok=True)

                for img in train_images:
                    img_path = os.path.join(images_dir, "train", img.name)
                    with open(img_path, "wb") as f:
                        f.write(img.getbuffer())

                for label in train_labels:
                    label_path = os.path.join(labels_dir, "train", label.name)
                    with open(label_path, "wb") as f:
                        f.write(label.getbuffer())

                for img in val_images:
                    img_path = os.path.join(images_dir, "val", img.name)
                    with open(img_path, "wb") as f:
                        f.write(img.getbuffer())

                for label in val_labels:
                    label_path = os.path.join(labels_dir, "val", label.name)
                    with open(label_path, "wb") as f:
                        f.write(label.getbuffer())

                # Save the YAML configuration
                save_yaml(nc, class_names)
                
                # Run the training command
                os.system(f"python /home/team04/yolov5/train.py --img 640 --batch 16 --epochs {epochs} --data /home/team04/yolov5/data/corning.yaml --weights yolov5s.pt --cache")
                st.success(f"Training completed for {epochs} epochs.")
            else:
                st.error("Please upload all required images and label files.")

# Function to get the latest exp directory in runs/train
def get_latest_exp_dir(base_path="/home/team04/yolov5/runs/train"):
    exp_dirs = [d for d in Path(base_path).iterdir() if d.is_dir() and d.name.startswith("exp")]
    latest_exp_dir = max(exp_dirs, key=lambda d: d.stat().st_mtime, default=None)
    return latest_exp_dir

with col3:
    st.header("Detect Image")
    
    st.markdown('<div class="column-container">', unsafe_allow_html=True)
    
    st.write('Find out where is the object')
    up_image = st.file_uploader("Choose a image file for classification")
    if up_image is not None:
        st.image(up_image)
        with open("/home/team04/yolov5/image.jpeg","wb") as f:
            f.write(up_image.getbuffer())

    st.write('Classification using YOLO')
    if st.button('Run YOLO Model'):
        if os.path.exists("/home/team04/yolov5/exp/"):
            shutil.rmtree("/home/team04/yolov5/exp/")
        
        latest_exp_dir = get_latest_exp_dir()
        if latest_exp_dir:
            weight_path = latest_exp_dir / "weights/best.pt"
            # st.write(f"Using weight path: {weight_path}")  # Display the weight path
            os.system(f"python /home/team04/yolov5/detect.py --weight {weight_path} --source /home/team04/yolov5/image.jpeg --project /home/team04/yolov5/ --save-txt --save-conf")
        
            latest_exp_dir = get_latest_exp_dir(base_path="/home/team04/yolov5")
            if latest_exp_dir and (latest_exp_dir / "image.jpeg").exists():
                st.image(str(latest_exp_dir / "image.jpeg"))
                label_file_path = latest_exp_dir / "labels/image.txt"
                if label_file_path.exists():
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        label_dict = {i: name for i, name in enumerate(class_names.split(','))}
                        results = []
                        for line in lines:
                            parts = line.strip().split()
                            label_index = int(parts[0])
                            label = label_dict.get(label_index, "Unknown")
                            confidence = float(parts[-1])
                            bbox = " ".join(parts[1:-1])
                            results.append({
                                "label": label,
                                "coordinates": bbox,
                                "confidence": confidence
                            })
                        for result in results:
                            st.write(f"Detection: {result['label']}")
                            st.write(f"Coordinates: {result['coordinates']}")
                            st.write(f"Accuracy: {result['confidence']:.2f}")

                else:
                    st.error("Label file not found. Please check the YOLOv5 output.")
            else:
                st.error("Detection image not found. Please check the YOLOv5 output.")
        else:
            st.error("No trained model found. Please check the YOLOv5 training output.")
    st.markdown('</div>', unsafe_allow_html=True)
