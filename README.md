# VerifyIt

An end-to-end machine learning project that trains a deep learning model to detect AI-generated Instagram reels, then wraps it in an interactive web game where players compete against the model.

🔗 **[Play the game here](https://verifyit-ai-or-real.streamlit.app)**

---

## What is VerifyIt?

VerifyIt is a browser-based game that shows you short video clips and asks one question: **is this AI-generated or real?** After you guess, the game reveals three things — what you said, what the AI model predicted (along with its confidence level), and the actual answer.

The game adapts to how well you're doing. Answer correctly and the next clip will be harder to identify. Get it wrong and the game eases up. You can play as many rounds as you want, and at the end you get a report showing how you performed against the AI.

---

## How It Was Built

The project has two parts: a machine learning pipeline and a web app.

**The ML pipeline** (in `notebook/model_training.ipynb`) extracts frames from 66 video clips, trains an EfficientNet-B0 classifier using transfer learning, optimizes the decision threshold, and scores every clip with a difficulty rating. The output is a single JSON file that powers the game. The notebook is heavily commented and walks through every step in detail.

**The web app** (in `game_app.py`) reads that JSON file and runs the game. It handles adaptive difficulty, score tracking, and streams videos from Google Drive. No model inference happens at runtime — everything was precomputed during training.

---

## Project Structure

```
VerifyIt/
├── game_app.py                 # Streamlit web app
├── clip_scores.json            # Precomputed model predictions and difficulty scores
├── requirements.txt            # Dependencies for Streamlit Cloud deployment
├── notebook/
│   ├── model_training.ipynb    # Full ML pipeline — frame extraction, training, evaluation, scoring
│   └── url_generation.ipynb    # Script to extract Google Drive file IDs and generate video URLs
```

---

## Tech Stack

**Model Training (Google Colab)**
- Python 3
- PyTorch and torchvision — model building, transfer learning, training loop
- EfficientNet-B0 — pretrained backbone for image classification
- OpenCV (cv2) — video file reading and frame extraction
- Pillow (PIL) — image processing and resizing
- scikit-learn — train/test splitting, classification metrics, confusion matrix
- matplotlib and seaborn — training curves and evaluation visualizations
- Google Drive API — extracting file IDs for video hosting

**Web App**
- Streamlit — game interface and deployment
- HTML/CSS — custom styling (Space Grotesk font, retro groovy theme)

**Hosting**
- Streamlit Community Cloud — app hosting
- Google Drive — video file hosting and streaming

---

## Run It Yourself

### Play the deployed game

No setup needed. Just visit the [live app](https://verifyit-ai-or-real.streamlit.app) in any browser.

### Run the game locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Offsideplayer24/VerifyIt.git
   cd VerifyIt
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   ```
   Activate it:
   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`

   Then install:
   ```bash
   pip install streamlit
   ```

3. Run the app:
   ```bash
   streamlit run game_app.py
   ```
   The game will open in your browser at `http://localhost:8501`.

### Retrain the model from scratch

If you want to run the full ML pipeline, open `notebook/model_training.ipynb` in [Google Colab](https://colab.research.google.com).

You will need:
1. The [REAL/AI Video Dataset](https://www.kaggle.com/datasets/kanzeus/realai-video-dataset) from Kaggle
2. A Google Drive folder with the clips organized into `ai/` and `real/` subfolders
3. The following Python libraries (all pre-installed in Colab):
   ```
   torch
   torchvision
   opencv-python-headless
   Pillow
   scikit-learn
   matplotlib
   seaborn
   ```
4. A GPU runtime is recommended (Runtime → Change runtime type → T4 GPU) but CPU works too — training takes about 15–20 minutes on CPU vs 3–5 minutes on GPU.

The notebook extracts frames, trains the model, optimizes the threshold, scores all clips, and saves `clip_scores.json` to Google Drive.

---

## Dataset

[REAL/AI Video Dataset](https://www.kaggle.com/datasets/kanzeus/realai-video-dataset) by kanzeus on Kaggle — 66 video clips (33 AI-generated, 33 real).

---

Built by Kaushal — Northeastern University, 2026