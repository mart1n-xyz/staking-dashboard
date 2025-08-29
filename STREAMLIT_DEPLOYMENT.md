# Streamlit Deployment Instructions

This document explains how to deploy the SNT Staking Dashboard to Streamlit Community Cloud using proper secrets management.

## Local Development Setup

1. Create the `.streamlit/secrets.toml` file (this file is already gitignored):
   ```toml
   [default]
   RPC_ENDPOINT = "https://public.sepolia.rpc.status.network"
   ```

2. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud Deployment

### Step 1: Deploy Your App

1. Push your code to GitHub (ensure `.streamlit/secrets.toml` is **not** included in the repository - it should be gitignored)
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Click "New app" and connect your GitHub repository
4. Select the repository and branch
5. Set the main file path to `app.py`

### Step 2: Configure Secrets

1. During deployment, click on "Advanced settings" 
2. In the "Secrets" section, paste the following:
   ```toml
   [default]
   RPC_ENDPOINT = "https://public.sepolia.rpc.status.network"
   ```
3. Click "Save" and then "Deploy"

### Step 3: Update Secrets (After Deployment)

If you need to update secrets after deployment:

1. Go to your app's dashboard in Streamlit Community Cloud
2. Click on the three dots menu (⋮) next to your app
3. Select "Settings"
4. Navigate to the "Secrets" tab
5. Update your secrets and click "Save"
6. Your app will automatically restart with the new secrets

## Important Notes

- The `.streamlit/secrets.toml` file should **NEVER** be committed to version control
- The app includes fallback logic to use environment variables (`.env`) for backward compatibility
- For production deployment, always use Streamlit's secrets management system
- The secrets are encrypted and securely stored by Streamlit Community Cloud

## Environment Variables vs Secrets

| Environment | Configuration Method |
|------------|---------------------|
| Local Development | `.streamlit/secrets.toml` (preferred) or `.env` (fallback) |
| Streamlit Cloud | Streamlit Secrets Management (Advanced Settings) |
| Other Deployment | Standard environment variables |

## Security Best Practices

1. ✅ Use Streamlit secrets for deployment
2. ✅ Keep `.streamlit/secrets.toml` in `.gitignore`
3. ✅ Never commit sensitive information to version control
4. ✅ Use the fallback environment variable system for local development if needed
5. ✅ Regularly rotate your RPC endpoints if using private/paid services

## Troubleshooting

### App can't find RPC_ENDPOINT
- **Local**: Check that `.streamlit/secrets.toml` exists and contains the correct configuration
- **Cloud**: Verify secrets are properly configured in Streamlit Cloud's Advanced Settings

### Secrets not updating
- Restart the app from the Streamlit Cloud dashboard after updating secrets
- Check that the TOML format is correct (no syntax errors)

### Local vs Cloud behavior differences
- The app will automatically try Streamlit secrets first, then fall back to environment variables
- This ensures compatibility across different deployment environments
