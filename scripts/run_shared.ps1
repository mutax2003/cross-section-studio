# Share on your local network (other devices use http://<your-ip>:8501)
Set-Location $PSScriptRoot\..
python -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
