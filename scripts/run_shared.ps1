# Share on your local network (other devices use http://<your-ip>:<port>)
param(
    [int]$Port = 18501
)

Set-Location $PSScriptRoot\..
Write-Host "Cross Section Studio on port $Port (LAN: http://<your-ip>:$Port)"
python -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port $Port
