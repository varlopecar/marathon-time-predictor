# Deployment Guide

This guide helps you deploy the Marathon Time Predictor API to various platforms.

## Pre-deployment Checklist

1. ✅ Replace `runtime.txt` with `.python-version` (already done)
2. ✅ Ensure all dependencies are in `requirements.txt`
3. ✅ Test locally with `python deploy_test.py`
4. ✅ Verify model file size (259MB - may need optimization)

## Platform-Specific Deployment

### Scalingo/Heroku

The application is configured for Scalingo deployment with:

- **Procfile**: `web: python startup.py`
- **Python version**: 3.11 (specified in `.python-version`)
- **Environment variables**: Configured in `scalingo.json`

#### Deployment Steps:

1. **Install Scalingo CLI** (if not already installed):

   ```bash
   npm install -g @scalingo/cli
   ```

2. **Login to Scalingo**:

   ```bash
   scalingo login
   ```

3. **Deploy the application**:

   ```bash
   scalingo create marathon-time-predictor
   scalingo git-set
   git push scalingo main
   ```

4. **Monitor deployment logs**:
   ```bash
   scalingo logs --follow
   ```

#### Troubleshooting:

- **Model loading timeout**: The 259MB model file may cause timeout issues
- **Memory limits**: Consider upgrading to a larger instance size
- **Startup time**: The application may take 30-60 seconds to start due to model loading

### Local Testing

Before deploying, test locally:

```bash
# Test deployment readiness
python deploy_test.py

# Test startup script
python startup.py

# Test API endpoints
curl http://localhost:8000/
curl http://localhost:8000/health
```

## Environment Variables

Key environment variables:

- `PORT`: Application port (default: 8000)
- `HOST`: Application host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: info)
- `ENVIRONMENT`: Environment name (default: production)

## Model Optimization

If deployment fails due to model size:

1. **Reduce model complexity**: Modify `n_estimators` in `RandomForestRegressor`
2. **Use model compression**: Consider using joblib instead of pickle
3. **Lazy loading**: Load model only when needed

## Health Checks

The application provides health check endpoints:

- `GET /`: Basic API information
- `GET /health`: Model status and health
- `POST /predict`: Prediction endpoint (requires model to be loaded)

## Monitoring

Monitor the application with:

```bash
# View logs
scalingo logs --follow

# Check application status
scalingo status

# Scale application if needed
scalingo scale web:1:S
```

## Common Issues

1. **Model loading timeout**: Increase startup timeout or optimize model
2. **Memory issues**: Upgrade instance size or optimize model
3. **Import errors**: Ensure all dependencies are in requirements.txt
4. **Port binding**: Ensure PORT environment variable is set correctly
