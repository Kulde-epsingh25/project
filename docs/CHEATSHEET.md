# Quick Start Cheat Sheet (Beginner)

Use this when you only want the exact steps quickly.

## 1. Open Project
- Open VS Code
- Open folder: `C:/Users/HP/Documents/NetBeansProjects/crypto_forecasting_project`

## 2. Setup Python Environment
In VS Code terminal (PowerShell):

```powershell
cd C:/Users/HP/Documents/NetBeansProjects/crypto_forecasting_project
py -3.11 -m venv .venv311
.venv311/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Train Models
```powershell
python src/train.py
```

## 4. Build Power BI Export Tables
```powershell
python src/build_powerbi_exports.py
```

## 5. Run Dashboard (Optional)
```powershell
python -m streamlit run src/dashboard.py --server.port 8503
```
Open: `http://localhost:8503`

## 6. Build Power BI Report
In Power BI Desktop:
1. Get Data > Text/CSV
2. Import all files from `outputs/powerbi`
3. Create Date table and relationships
4. Paste DAX from `docs/POWERBI_MEASURES.dax`
5. Build visuals from `docs/POWERBI_VISUAL_FIELD_MAP.md`

## Important Files
- Full beginner guide: `docs/BEGINNER_SETUP_GUIDE.md`
- Page mapping: `docs/POWERBI_BUILD_GUIDE.md`
- Visual mapping: `docs/POWERBI_VISUAL_FIELD_MAP.md`
- DAX pack: `docs/POWERBI_MEASURES.dax`

## If Something Fails
- TensorFlow issue: ensure Python 3.11 is used.
- Port busy: change streamlit port to `8504` or `8505`.
- Activation blocked: run PowerShell as admin once:
```powershell
Set-ExecutionPolicy RemoteSigned
```
