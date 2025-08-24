# 🚀 Honeywell Anomaly Detection Dashboard - Two-Tab System

## 📋 Overview

This dashboard provides a comprehensive two-tab interface for analyzing anomaly detection results:

1. **📊 Original Dataset Analysis Tab**: Always displays results from the original TEP dataset
2. **🔬 Custom Dataset Analysis Tab**: Allows users to upload and analyze their own datasets

## 🎯 Features

### Tab 1: Original Dataset Analysis
- **Always Available**: Results from `TEP_Output_Scored.csv` are always displayed
- **Project Information**: Comprehensive details about the Honeywell anomaly detection methodology
- **Training Validation**: Automatic validation of training period requirements
- **Visualizations**: Timeline charts, severity distributions, and feature analysis
- **Top Anomalies**: Detailed table of the most significant anomalies

### Tab 2: Custom Dataset Analysis
- **Raw Data Processing**: Upload raw CSV files and process them through the anomaly detection pipeline
- **Pre-scored Data**: Upload already-processed CSV files for immediate analysis
- **Same Visualizations**: Identical analysis capabilities as the original dataset
- **Export Options**: Download results and summaries

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required packages (see `requirements.txt`)
- TEP_Output_Scored.csv file in the project directory

### Installation
```bash
# Clone or download the project
cd honeywell_anomaly_project

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

### Access the Dashboard
Open your browser and navigate to: `http://localhost:8501`

## 📊 Using Tab 1: Original Dataset Analysis

This tab automatically loads and displays:
- **Dataset Metrics**: Total rows, features, date range, max score
- **Training Validation**: Checks if training period meets requirements (mean < 10, max < 25)
- **Anomaly Timeline**: Complete timeline with training period highlighted
- **Severity Distribution**: Breakdown of anomaly severity levels
- **Top Anomalies**: Table of the 50 most significant anomalies

## 🔬 Using Tab 2: Custom Dataset Analysis

### Option 1: Raw Data Processing

1. **Select Upload Type**: Choose "Raw CSV (will be processed)"
2. **Upload File**: Select your raw CSV file
3. **Configure Parameters**:
   - **Time Column Name**: Specify your timestamp column
   - **Training Start/End**: Define the normal behavior period
   - **Analysis End**: Set the end of analysis period
4. **Processing Options**:
   - **Apply Smoothing**: Reduce noise in anomaly scores
   - **Save Report**: Generate PDF reports
5. **Process**: Click "🚀 Process Dataset" to run the pipeline
6. **View Results**: Same visualizations as the original dataset

### Option 2: Pre-scored Data

1. **Select Upload Type**: Choose "Scored CSV (already processed)"
2. **Upload File**: Select your pre-scored CSV file
3. **Immediate Analysis**: View results instantly with the same visualizations

### Required CSV Format

**For Raw Data:**
- CSV file with timestamp column
- Numeric feature columns
- Regular time intervals (hourly recommended)

**For Scored Data:**
- `Time` (or your timestamp column)
- `Abnormality_score` (0-100 values)
- `top_feature_1` through `top_feature_7`

## 🔧 Processing Pipeline

When processing raw data, the dashboard automatically:
1. **Loads** your CSV file
2. **Validates** time column and data format
3. **Runs** the anomaly detection pipeline (`main.py`)
4. **Generates** scored output with the same format as the original dataset
5. **Displays** comprehensive analysis results

## 📈 Visualizations Available

Both tabs provide identical visualization capabilities:

### Timeline Analysis
- **Anomaly Score Timeline**: Complete time series with training period highlighted
- **Severity Thresholds**: Visual markers for different severity levels
- **Interactive Hover**: Detailed information on hover

### Statistical Analysis
- **Severity Distribution**: Bar charts showing anomaly severity breakdown
- **Training Validation**: Metrics and status indicators
- **Feature Analysis**: Top contributing factors for severe anomalies

### Data Tables
- **Top Anomalies**: Sortable table of most significant anomalies
- **Feature Contributions**: Detailed breakdown of contributing factors
- **Export Options**: Download results in CSV format

## 💾 Export Capabilities

### Custom Dataset Results
- **Full Results CSV**: Complete processed dataset
- **Severity Summary**: Statistical breakdown by severity level

### Original Dataset
- **Always Available**: No export needed - always visible
- **Reference Data**: Use as baseline for comparison

## 🎯 Use Cases

### Industrial Process Monitoring
- Compare new sensor data with established baselines
- Identify process drift and anomalies
- Validate training periods for new equipment

### Research and Development
- Test anomaly detection on new datasets
- Compare different training periods
- Analyze feature contributions across datasets

### Quality Assurance
- Validate anomaly detection models
- Ensure consistent scoring across datasets
- Generate reports for stakeholders

## 🔍 Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Ensure `TEP_Output_Scored.csv` is in the project directory
   - Check file permissions and paths

2. **Processing Failures**
   - Verify CSV format and column names
   - Check that time column contains valid timestamps
   - Ensure sufficient training data (>72 rows)

3. **Visualization Issues**
   - Check browser compatibility
   - Ensure all required packages are installed
   - Verify data format in uploaded files

### Performance Tips

- **Large Datasets**: Consider sampling for initial exploration
- **Processing**: Raw data processing may take several minutes for large files
- **Memory**: Close other applications if processing very large datasets

## 📚 Technical Details

### Architecture
- **Frontend**: Streamlit web application
- **Backend**: Python anomaly detection pipeline
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Plotly for interactive charts

### Data Flow
1. **Input**: Raw CSV or pre-scored CSV
2. **Processing**: Anomaly detection pipeline (if raw data)
3. **Analysis**: Statistical analysis and visualization
4. **Output**: Interactive dashboard with export options

### File Structure
```
honeywell_anomaly_project/
├── dashboard.py              # Main dashboard application
├── main.py                   # Anomaly detection pipeline
├── TEP_Output_Scored.csv    # Original dataset results
├── pipeline/                 # Processing modules
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # Anomaly detection models
│   ├── scoring.py           # Score calculation and calibration
│   └── io.py                # Input/output operations
└── requirements.txt          # Python dependencies
```

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure data format matches requirements
4. Check console output for error messages

## 📄 License

This project implements Honeywell's multivariate anomaly detection algorithm for industrial process monitoring.

---

**🔍 Built with Streamlit • Powered by Python • Industrial-Grade Anomaly Detection**
