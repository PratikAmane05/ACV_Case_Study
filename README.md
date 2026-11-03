# ACV_Case_Study

# Advanced Computer Vision – CA-II Case Studies

This repository contains the practical implementation of two case studies for the **Advanced Computer Vision (ACV)** subject. The experiments demonstrate feature matching using SIFT and KNN, and vehicle detection in traffic videos using a pretrained YOLO model.

## 📌 Project Overview

The assignment consists of two computer vision applications:

1. **Image Recognition Using Feature Matching** – Identifying the same book cover in images captured with different perspectives using SIFT, KNN matching and RANSAC.
2. **Vehicle Detection Using YOLO** – Detecting and classifying vehicles in traffic videos using a pretrained YOLO26n model.

---

# Case Study 1: Image Recognition Using Feature Matching

## 🎯 Objective

To identify whether two images represent the same book cover, even when they have differences in viewing angle and perspective.

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- SIFT (Scale-Invariant Feature Transform)
- KNN (K-Nearest Neighbors)
- RANSAC (Random Sample Consensus)

## ⚙️ Implementation

The experiment was performed using a public-domain *The Jungle Book* cover image.

### Step 1: Image Acquisition
- Downloaded the reference book-cover image.
- Created a second image using perspective transformation.
- Applied a slight brightness adjustment to simulate image variation.

### Step 2: SIFT Feature Extraction
- Converted both images into grayscale.
- Detected distinctive keypoints using SIFT.
- Generated descriptors for the detected keypoints.
- Visualized the extracted features.

### Step 3: KNN Feature Matching
- Used Brute Force Matcher with the Euclidean (L2) distance metric.
- Applied KNN with `k = 2`.
- Used Lowe's Ratio Test with a threshold of `0.75` to filter weak matches.

### Step 4: RANSAC Verification
- Applied RANSAC to estimate the homography between the two images.
- Separated geometrically consistent matches (inliers) from incorrect matches (outliers).
- Visualized the final matching results.

## 📊 Experimental Results

| Performance Metric | Result |
|---|---:|
| Total KNN Matches | 1,366 |
| Good Matches After Ratio Test | 553 |
| RANSAC Inliers | 527 |
| RANSAC Outliers | 26 |
| Inlier Percentage | **95.30%** |

**Inlier Percentage Formula:**

`(RANSAC Inliers / Good Matches) × 100`

`(527 / 553) × 100 = 95.30%`

## 📝 Analysis

The SIFT algorithm successfully extracted distinctive features from both book-cover images. KNN matching identified 553 good matches, while RANSAC retained 527 geometrically consistent matches.

The 95.30% inlier percentage indicates strong feature correspondence between the original and perspective-transformed images.

## ⚠️ Limitations

- Performance may decrease under extreme perspective changes.
- Blurred or low-resolution images can affect feature extraction.
- Repeated patterns may produce incorrect matches.
- Significant occlusion and lighting changes may reduce matching performance.
- The perspective variation in this experiment was simulated rather than captured using a real camera.

---

# Case Study 2: Vehicle Detection Using YOLO

## 🎯 Objective

To detect and classify vehicles in traffic videos using a pretrained YOLO model and evaluate its performance under different traffic conditions.

## 🛠️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLO26n
- NumPy
- Matplotlib
- Pandas

## ⚙️ Implementation

A public traffic video was used to demonstrate vehicle detection.

### Step 1: Model Setup
- Installed the Ultralytics library.
- Loaded the pretrained `yolo26n.pt` model.
- Used the pretrained model for object detection.

### Step 2: Video Acquisition
- Downloaded a public traffic video.
- Loaded the video using OpenCV.
- Extracted video frames for processing.

### Step 3: Vehicle Detection
- Applied YOLO26n to the video frames.
- Detected vehicles using bounding boxes.
- Displayed class labels and confidence scores.
- Focused on four vehicle categories:
  - Car
  - Motorcycle
  - Bus
  - Truck

### Step 4: Video Processing
- Processed all 251 video frames.
- Generated bounding boxes and class labels for detected vehicles.
- Saved the annotated video as `vehicle_detection_output.mp4`.

### Step 5: Performance Evaluation
Evaluated three selected frames:
- Beginning – Frame 0
- Middle – Frame 125
- Late Traffic – Frame 200

## 📊 Vehicle Detection Results

### Overall Detection Counts

The following counts were accumulated across all 251 processed frames.

| Vehicle Category | Detection Count |
|---|---:|
| Cars | 2,256 |
| Motorcycles | 101 |
| Buses | 0 |
| Trucks | 391 |
| **Total Detections** | **2,748** |

*Note: These are frame-level detection counts, not unique vehicle counts. A vehicle appearing in multiple frames can be counted multiple times.*

### Sample-Based Evaluation

| Metric | Beginning | Middle | Late Traffic |
|---|---:|---:|---:|
| Frame Number | 0 | 125 | 200 |
| Cars | 8 | 7 | 9 |
| Motorcycles | 0 | 1 | 1 |
| Buses | 0 | 0 | 0 |
| Trucks | 1 | 2 | 2 |
| Total Vehicles | **9** | **10** | **12** |
| Average Confidence | 0.576 | 0.505 | 0.581 |

## 📝 Analysis

The YOLO26n model successfully detected vehicles in the selected traffic frames and generated bounding boxes with class labels and confidence scores.

The model detected 9 vehicles in the beginning frame, 10 in the middle frame and 12 in the late traffic frame.

The average confidence scores were 0.576, 0.505 and 0.581, respectively. These results demonstrate the model's ability to identify different vehicle categories in the tested traffic scenes.

## ⚠️ Limitations

- Small and distant vehicles may be missed.
- Overlapping vehicles can affect detection.
- Detection confidence may vary with lighting and visibility.
- Incorrect object classification is possible.
- The experiment does not establish formal accuracy, precision or recall because ground-truth annotations were not available.
- Only three selected frames were used for the sample-based evaluation.

---

# 📁 Repository Structure

```text
Advanced-Computer-Vision-CA-II/
│
├── README.md
│
├── Case_Study_1/
│   └── ACV_CA2_Case_Study_1.ipynb
│
├── Case_Study_2/
│   ├── ACV_CA2_Case_Study_2_YOLO.ipynb
│   └── vehicle_detection_output.mp4
│
└── Screenshots/
    ├── sift_features.png
    ├── knn_matching.png
    ├── ransac_matching.png
    ├── sift_performance.png
    ├── yolo_detection.png
    └── traffic_evaluation.png
```

*The structure above is the recommended organization for the repository. Add the corresponding files and screenshots before using it as the final repository structure.*

# 🚀 How to Run the Project

### Requirements

- Python 3.x
- Google Colab or Jupyter Notebook
- Internet connection for downloading images, video and pretrained model weights

### Install Dependencies

```bash
pip install opencv-contrib-python numpy matplotlib pandas ultralytics
```

### Running Case Study 1

1. Open `ACV_CA2_Case_Study_1.ipynb` in Google Colab.
2. Install the required dependencies.
3. Run the notebook cells sequentially.
4. Observe SIFT keypoints, KNN matches and RANSAC results.

### Running Case Study 2

1. Open `ACV_CA2_Case_Study_2_YOLO.ipynb` in Google Colab.
2. Install Ultralytics.
3. Load the pretrained YOLO26n model.
4. Download the traffic video.
5. Execute the detection and evaluation cells.
6. View and save the annotated output video.

# 📚 Learning Outcomes

Through these case studies, we explored:

- Feature extraction and image representation.
- Scale-Invariant Feature Transform (SIFT).
- KNN-based feature matching.
- Lowe's Ratio Test.
- Homography estimation using RANSAC.
- Object detection using pretrained YOLO models.
- Video frame processing using OpenCV.
- Vehicle classification and confidence score analysis.
- Practical limitations of computer vision systems.

# 🏁 Conclusion

Both case studies demonstrate practical applications of Advanced Computer Vision.

The first case study uses SIFT, KNN and RANSAC to identify matching features between images with different perspectives. The second case study uses YOLO26n to detect and classify vehicles in traffic videos.

These experiments provided practical understanding of image feature matching, object detection, video processing and performance evaluation using Python and OpenCV.

---

**Subject:** Advanced Computer Vision (ACV)  
**Assignment:** CA-II Case Studies  
**Academic Year:** 2026–2027  
**Implementation:** Python and Google Colab
