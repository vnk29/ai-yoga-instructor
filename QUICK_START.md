# Quick Start Guide

## Running the Application

### 1. Open Terminal/PowerShell in the project folder
```powershell
cd "c:\Users\Keerthana\AI YOGA INSTRUCTOR\plank-posture"
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\activate
```

### 3. Start the Streamlit App
```powershell
streamlit run app.py
```

The app will automatically open in your browser at: **http://localhost:8501**

---

## Optional: Process Dataset for Better Detection

```powershell
# Auto-detect dataset in common locations
python setup_dataset.py

# Or specify dataset path explicitly
python setup_dataset.py --dataset-dir "C:\path\to\dataset"
```

---

## Features Available

| Feature | How to Access |
|---------|--------------|
| **Live Yoga Mode** | Sidebar → "Live Yoga Mode" → Click "Start AI Coaching" |
| **Upload Image** | Sidebar → "Upload Image Mode" → Choose image file |
| **Get Recommendations** | Sidebar → "Yoga Recommendation Mode" → Select goals |
| **View About** | Sidebar → "About Project" |

---

## Troubleshooting

**Issue: "streamlit command not found"**
- Solution: Ensure virtual environment is activated (see step 2 above)

**Issue: "No webcam detected"**
- Solution: Check camera is connected and in use
- Or modify `CAMERA_INDEX = 0` in `yoga_config.py`

**Issue: "Image upload fails"**
- Solution: Try a different image format (JPG/PNG)
- Ensure image has good lighting and full body visible

---

## File Locations

- **App:**           `app.py`
- **Config:**        `yoga_config.py`
- **Dataset Setup:** `setup_dataset.py`
- **Logs:**          `session_log.json`
- **Data Folder:**   `data/` (contains pose statistics)

---

**Happy Yoga Practice! 🧘**
