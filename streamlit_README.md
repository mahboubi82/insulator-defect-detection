# Insulator Defect Detection — Web App

YOLOv8-powered web app for detecting and classifying electrical insulator defects.

**Deployed on:** Streamlit Cloud  
**Model:** YOLOv8s fine-tuned on NBPower insulator dataset  
**Classes:** `broken` · `insulator` · `pollution-flashover`

---

## Repo structure

```
├── app.py                  ← main Streamlit app
├── best.pt                 ← YOLOv8 trained weights  ← YOU MUST ADD THIS
├── requirements.txt        ← dependencies
└── .streamlit/
    └── config.toml         ← theme settings
```

---

## Deploy to Streamlit Cloud

1. Create a GitHub repo and push all files
2. Add your `best.pt` model weights to the repo root
3. Go to share.streamlit.io → New app → select your repo
4. Main file: `app.py`
5. Click Deploy

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Usage

1. Open the web app URL
2. Upload any isolator image (jpg, png, bmp)
3. Adjust confidence threshold in sidebar if needed
4. See bounding boxes + class labels on the image
5. Download the annotated result
