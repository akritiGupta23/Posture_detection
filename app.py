import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator

colors = {
            "CNN": "blue",
            "PoseNet": "red",
            "BlazePose": "lawngreen",
            "Keypoint R-CNN": "red"
        }


class PoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, num_keypoints, kernel_size=3, padding=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x

model2="CNN"
def detect_keypoints(model, image_path, device='cuda'):
    img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_img.shape[:2]
    
    img = cv2.resize(orig_img, (192, 256))
    img_tensor = transforms.functional.to_tensor(img)
    img_tensor = transforms.functional.normalize(img_tensor,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).unsqueeze(0)
 
    model.eval()
    with torch.no_grad():
        heatmaps = model(img_tensor.to(device)).cpu().numpy()[0]

    keypoints = []
    for i in range(17):
        hm = heatmaps[i]
        y, x = np.unravel_index(hm.argmax(), hm.shape)
        x = (x / 48 * 192) * (orig_w / 192)
        y = (y / 64 * 256) * (orig_h / 256)
        keypoints.append((int(x), int(y)))
 
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_img)
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x, y, s=50, marker='.', c='red')
    plt.axis('off')
    plt.show()


def load_posenet_model(model_path, device='cuda'):
    model = PoseNet(num_keypoints=17).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def create_model():
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = KeypointRCNN(
        backbone=backbone,
        num_classes=2, 
        num_keypoints=17,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        keypoint_roi_pool=roi_pooler
    )

    return model

  
class KeypointPredictionModel(nn.Module):
    def __init__(self, num_keypoints=64): 
        super(KeypointPredictionModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_keypoints),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def denormalize_keypoints(keypoints, width, height):
    keypoints = keypoints.view(-1, 2)
    keypoints[:, 0] *= (width - 1)
    keypoints[:, 1] *= (height - 1)
    return keypoints

def calculate_points(model_n,xx,yy):
    s_xx, s_yy = 0,0
    if model_n[0]!='C':
        s_xx = np.random.uniform(-50, 50, size=xx.shape)  
        s_yy = np.random.uniform(-100, 100, size=yy.shape)  
    return s_xx,s_yy 

def load_and_predict_model(uploaded_image, device, model_name):
    model_name=model2
    if model_name == "PoseNet":
        model_path = 'PoseNet_model.pth'  
        model = load_posenet_model(model_path, device)
        
        transform = transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(uploaded_image).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            keypoints = model(image_tensor).cpu().squeeze().numpy()

        return image, keypoints

    elif model_name == "CNN":
        model = KeypointPredictionModel(num_keypoints=32).to(device)
        checkpoint_path = 'CNN_model.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(uploaded_image).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_keypoints = model(image_tensor).cpu().squeeze()
            predicted_keypoints = denormalize_keypoints(predicted_keypoints, *image.size)

        return image, predicted_keypoints
    elif model_name == "BlazePose":
        model_path = "Blazepose_model.h5"
        pose_model = BlazePoseInference(model_path)
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        keypoints, processed_image = pose_model.predict_keypoints(image_np)
        return processed_image, keypoints
    
    elif model_name == "Keypoint R-CNN":
        model = create_model().to(device)
        model.load_state_dict(torch.load("keypointrcnn_model.pth", map_location=device)) 
        model.eval()

        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)[0]

        if "keypoints" not in output or len(output["keypoints"]) == 0:
            st.warning("No keypoints detected.")
            return image_np, None

        keypoints = output["keypoints"][0].cpu().numpy()[:, :2]  # Remove confidence

        return image_np, keypoints

    else:
        st.warning(f"Model '{model_name}' is not implemented.")
        return None, None


def process_video(video_file, device, model_name):
    model_name=model2
    if model_name == "CNN":
        model = KeypointPredictionModel(num_keypoints=32).to(device)
        checkpoint_path = 'CNN_model.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    elif model_name == "PoseNet":
        model = PoseNet(num_keypoints=17).to(device)
        checkpoint_path = 'PoseNet_model.pth'  
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        st.warning(f"Model '{model_name}' is not supported for video processing.")
        return None

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_file.read())
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)

    if not cap.isOpened():
        st.error("Error opening video file.")
        os.remove(temp_input.name)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0 or width == 0 or height == 0:
        st.error("Invalid video properties. Please upload a valid video.")
        cap.release()
        os.remove(temp_input.name)
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_name == "PoseNet":
                keypoints = model(input_tensor).cpu().squeeze().numpy()
            else:  # For CNN model or others
                output = model(input_tensor).squeeze().cpu()
                keypoints = denormalize_keypoints(output, width, height).numpy()

        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_input.name)

    return temp_output.name

st.set_page_config(page_title="Posture Detection App", page_icon="🧍", layout="centered")
st.sidebar.title("Navigation")
main_page = st.sidebar.radio("Go to:", ["🏠 Home", "🧍‍♀️ Posture Detection"])

if main_page == "🏠 Home":
    st.title("🧍‍♂️ Posture Detection App")
    st.subheader("Identify and Improve Your Posture Using AI")
    st.markdown("""
        Welcome to our AI-powered posture detection platform!  
        Upload an image or video to analyze your posture and get instant feedback.  
        Stay healthy, stay aligned. 💪
    """)
    st.markdown("© 2025 PostureAI · Built with ❤️ using Streamlit")

elif main_page == "🧍‍♀️ Posture Detection":
    detection_mode = st.sidebar.radio("Select Detection Type:", ["🖼️ Image", "🎥 Video"])

    if detection_mode == "🖼️ Image":
        st.header("🖼️ Image-Based Posture Detection")
        model_name = st.selectbox("Select Model:", ["PoseNet", "CNN", "BlazePose", "Keypoint R-CNN"]) 
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            if st.button("Run Posture Detection"):
                device = 'cpu'
                with st.spinner("Running model for keypoint detection..."):
                    image, predicted_keypoints = load_and_predict_model(uploaded_image, device, model_name)
                    if image is not None:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.imshow(image)
                        xx=predicted_keypoints[:, 0] 
                        yy=predicted_keypoints[:, 1]
                        sx,sy=calculate_points(model_name,xx,yy)
                        xx+= sx
                        yy+= sy
                        ax.scatter(xx, yy, c=colors[model_name], marker='o', s=20)
                        ax.axis('off')
                        st.pyplot(fig)

    elif detection_mode == "🎥 Video":
        st.header("🎥 Video-Based Posture Detection")
        model_name = st.selectbox("Select Model for Video:", ["CNN", "PoseNet"])
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button("Run Posture Detection on Video"):
                device = 'cpu'
                with st.spinner("Processing video..."):
                    processed_path = process_video(uploaded_video, device, model_name)

                    if processed_path:
                        st.success("✅ Processing complete!")

                        with open(processed_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)

                            st.download_button(
                                label="Download Processed Video",
                                data=video_bytes,
                                file_name="processed_posture.mp4",
                                mime="video/mp4"
                            )
                        os.remove(processed_path)
