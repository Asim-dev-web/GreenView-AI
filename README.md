# 🌍 GreenView AI: RGB Satellite Vegetation Segmentation

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20Live%20App-yellow)](https://huggingface.co/spaces/As-im/GreenView-AI)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Download%20Dataset-blue)](https://www.kaggle.com/datasets/asim3000/satellite-imagery-urban-tree-segmentation-india)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io)

**GreenView AI** is a deep learning model designed to detect and map vegetation—including trees, agricultural land, and dense foliage—using only standard RGB satellite imagery.

Unlike traditional methods that rely on Infrared/NDVI data (which is often low-resolution or expensive), GreenView AI uses **U-Net++** to identify greenery based on texture and context from high-resolution Google Earth snapshots.

---

## **Try it Live**
No installation needed. Run the model in your browser here:
### [Launch GreenView AI on Hugging Face](https://huggingface.co/spaces/As-im/GreenView-AI)

---

## **The Challenge: Why "Just Green" Isn't Enough**

My goal was to build an AI that works on **standard high-res satellite images** (like Google Maps/Earth) which only have Red, Green, and Blue channels. This creates massive engineering hurdles that simple "Color Detection" cannot solve:

### 1. The "Green Trap" (RGB vs NDVI)
* **The Problem:** Without an Infrared channel (NDVI), a simple code script cannot tell the difference between a **Tree** and a **Green Rooftop**, **Algae in a Pond**, or **Green Astro-turf**.
* **The Solution:** I trained a Deep Learning model (U-Net++) to learn **texture patterns**, distinguishing rough, irregular tree canopies from smooth, geometric man-made structures.

### 2. The "Lazy Labeling" Problem (Dataset Bias)
* **The Issue:** Public datasets like **LoveDA** are massive but flawed. They suffer from:
    1.  **Lazy Labeling:** Marking entire forest blocks as one polygon (ignoring gaps).
    2.  **Domain Shift:** Being trained on Chinese cities, they failed to recognize the dense, unstructured vegetation patterns typical of Indian cities.
* **The Result:** Models trained *only* on LoveDA missed trees in dense clusters and failed on Indian architecture.

---

## **The Solution: Transfer Learning & Hybrid Training**

To solve the accuracy issues, I implemented a **Two-Stage Training Pipeline**:

1.  **Stage 1: Pre-Training (General Knowledge)**
    * I first trained the **U-Net++ (ResNet34)** model on the large-scale **LoveDA dataset**. This taught the model high-level urban features (roads, buildings, water).
2.  **Stage 2: Fine-Tuning (The "Indian Context")**
    * To fix the regional bias, I created a **custom dataset of 87 high-res image chips** centered on **Connaught Place (CP), New Delhi**.
    * I then **fine-tuned** the model on a **hybrid mix** of the LoveDA data and my custom CP dataset.
    * *Result:* This "mixed" approach allowed the model to retain its general understanding of cities while learning specifically to detect Indian tree varieties and ignore local noise (like complex shadow patterns).

---

## **Dataset Details**

The model uses a combination of open-source and custom-curated data:

| Dataset Source | Purpose | Details |
| :--- | :--- | :--- |
| **LoveDA (Urban)** | **Pre-training** | Large-scale dataset for general urban parsing. |
| **Custom (CP-India)** | **Fine-tuning** | 87 Manually annotated pairs (512x512) from New Delhi. |

**[📂 Download the Custom CP Dataset on Kaggle](https://www.kaggle.com/datasets/asim3000/satellite-imagery-urban-tree-segmentation-india)**

* **Annotations:** Binary Segmentation
    * `0` (Black) = Background (Buildings, Roads, Water, Shadows)
    * `255` (White) = Vegetation (Trees, Agriculture, Shrubs)

---

## **Tech Stack**

* **Deep Learning:** PyTorch, Segmentation Models PyTorch (SMP)
* **Model Architecture:** U-Net++ (ResNet34 Backbone)
* **Training Strategy:** Transfer Learning (Pre-trained on ImageNet -> LoveDA -> Custom Mix)
* **App Framework:** Streamlit (Python)
* **Deployment:** Hugging Face Spaces (Docker Container)
* **CI/CD:** GitHub Actions (Automated Model Deployment)

---

## **Local Installation**

To run this project on your own machine:

```bash
# 1. Clone the repository
git clone [https://github.com/As-im/GreenView-AI.git](https://github.com/As-im/GreenView-AI.git)
cd GreenView-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the App
streamlit run app.py

# 3. Run the App
streamlit run app.py
