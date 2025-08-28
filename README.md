# SNT Staking Dashboard

A Streamlit-based dashboard for monitoring and managing SNT staking activities.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp env.example .env
# Edit .env and add your RPC endpoint
```

3. Run the dashboard:
```bash
streamlit run app.py
```

4. Open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## Features

- Clean, sidebar-free interface
- Staking metrics overview
- Real-time staking data visualization
- Staking action interface

## Development

The main application file is `app.py`. Add your staking logic, data fetching, and visualizations there.

Built with ❤️ using Streamlit
