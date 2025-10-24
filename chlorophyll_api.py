import os
import ee
import numpy as np
import datetime
import joblib
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google Earth Engine credentials
GOOGLE_CREDENTIALS = "secrets/ee-ulikhasanah16-02d7612d178d.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS

try:
    service_account = "ee-ulikhasanah16-02d7612d178d@developer.gserviceaccount.com"
    credentials = ee.ServiceAccountCredentials(service_account, GOOGLE_CREDENTIALS)
    ee.Initialize(credentials)
    logger.info("Google Earth Engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Earth Engine: {e}")
    raise

# Load model and scaler
try:
    catboost_model = joblib.load("catboost_chlor_a.pkl")
    scaler = joblib.load("scaler.pkl")
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {e}")
    raise

# API configuration
app = FastAPI(
    title="Chlorophyll-a Prediction API",
    description="Predict chlorophyll-a concentration using satellite data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chlorophyll-a.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model definitions
class Location(BaseModel):
    lat: float
    lon: float
    date: str = None

class PredictionResponse(BaseModel):
    lat: float
    lon: float
    chlorophyll_a: float
    used_satellite: str
    dates: dict
    sst_date: str = None

# Satellite and band configuration
SATELLITES = {
    "Sentinel-2": "COPERNICUS/S2_SR",
    "VIIRS": "NOAA/VIIRS/001/VNP09GA"
}

BANDS = {
    "Sentinel-2": {
        "Red": "B4", "NIR": "B8", "SWIR1": "B11", 
        "SWIR2": "B12", "Blue": "B2", "Green": "B3"
    },
    "VIIRS": {
        "Red": "M5", "NIR": "M7", "SWIR1": "M10", 
        "SWIR2": "M11", "Blue": "M3", "Green": "M4"
    }
}

def get_rectangle_area(lat: float, lon: float, pixel_size: int = 30, pixel_count: int = 3):
    """Create a 3x3 pixel area (approximately 90x90 meters)"""
    offset = (pixel_size * pixel_count / 2) / 111320  # convert meters to degrees
    return ee.Geometry.Rectangle([
        lon - offset, lat - offset, 
        lon + offset, lat + offset
    ])

def get_satellite_data(lat: float, lon: float, collection_id: str, bands: dict, target_date: str):
    """Get satellite imagery data from the specified area"""
    try:
        area = get_rectangle_area(lat, lon)
        start_date = ee.Date(target_date) if target_date else ee.Date(
            datetime.datetime.utcnow().strftime('%Y-%m-%d')
        )
        
        collection = (ee.ImageCollection(collection_id)
                     .filterBounds(area)
                     .filterDate(start_date.advance(-30, 'day'), 
                               start_date.advance(30, 'day')))
        
        def add_diff(image):
            return image.set("date_diff", 
                           ee.Number(image.date().difference(start_date, "day")).abs())
        
        collection = collection.map(add_diff).sort("date_diff")
        image = collection.first()
        
        if image is not None and image.getInfo():
            date_info = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            data = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area,
                scale=30,
                maxPixels=1e9
            ).getInfo()
            
            return {band: data.get(bands[band], None) for band in bands}, date_info
            
    except Exception as e:
        logger.error(f"Failed to retrieve satellite data: {e}")
        
    return None, None

def get_sst_data(lat: float, lon: float, target_date: str):
    """Get Sea Surface Temperature data"""
    try:
        area = get_rectangle_area(lat, lon, pixel_size=5000)  # SST pixel size ~5 km
        start_date = ee.Date(target_date) if target_date else ee.Date(
            datetime.datetime.utcnow().strftime('%Y-%m-%d')
        )
        
        collection = (ee.ImageCollection("NOAA/CDR/OISST/V2_1")
                     .filterBounds(area)
                     .filterDate(start_date, start_date.advance(1, 'day'))
                     .sort("system:time_start"))
        
        image = collection.first()
        
        if image is not None and image.getInfo():
            date_info = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            data = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area,
                scale=5000,
                maxPixels=1e9
            ).getInfo()
            
            return data.get("sst", None), date_info
            
    except Exception as e:
        logger.error(f"Failed to retrieve SST data: {e}")
        
    return None, None

def calculate_ndci(red: float, nir: float):
    """Calculate Normalized Difference Chlorophyll Index"""
    if red is None or nir is None:
        return None
    return (nir - red) / (nir + red) if (nir + red) != 0 else None

def process_request(lat: float, lon: float, date: str = None):
    """Main function to process chlorophyll-a prediction request"""
    data_sources, dates = {}, {}
    
    # Get satellite data
    for sat in SATELLITES:
        data, date_info = get_satellite_data(lat, lon, SATELLITES[sat], BANDS[sat], date)
        if data:
            data_sources[sat] = data
            dates[sat] = date_info
    
    # Get SST data
    sst_value, sst_date = get_sst_data(lat, lon, date)
    
    # Select satellite (prefer Sentinel-2, fallback to VIIRS)
    selected_sat = "Sentinel-2" if "Sentinel-2" in data_sources else "VIIRS" if "VIIRS" in data_sources else None
    
    if not selected_sat:
        raise HTTPException(
            status_code=404, 
            detail=f"No satellite data available for location ({lat}, {lon}) on {date}"
        )
    
    selected_data = data_sources[selected_sat]
    band_mapping = BANDS[selected_sat]
    
    # Prepare features
    features = {
        "latitude": lat,
        "longitude": lon,
        "SWIR1": selected_data.get(band_mapping["SWIR1"]),
        "SWIR2": selected_data.get(band_mapping["SWIR2"]),
        "Blue": selected_data.get(band_mapping["Blue"]),
        "Green": selected_data.get(band_mapping["Green"]),
        "Red": selected_data.get(band_mapping["Red"]),
        "NIR": selected_data.get(band_mapping["NIR"]),
        "sst": sst_value if sst_value is not None else 0
    }
    
    # Add temporal features
    date_obj = datetime.datetime.strptime(dates[selected_sat], "%Y-%m-%d")
    features["dayofyear"] = date_obj.timetuple().tm_yday
    features["day_sin"] = np.sin(2 * np.pi * features["dayofyear"] / 365)
    features["day_cos"] = np.cos(2 * np.pi * features["dayofyear"] / 365)
    
    # Calculate NDCI
    features["NDCI"] = calculate_ndci(features["Red"], features["NIR"])
    
    # Prepare input for model
    input_data = np.array([[features[f] for f in features.keys()]])
    normalized_input = scaler.transform(input_data)
    
    # Make prediction
    chl_a_prediction = catboost_model.predict(normalized_input)[0]
    
    return {
        "lat": lat,
        "lon": lon,
        "chlorophyll_a": chl_a_prediction * 1000,  # Convert to mg/m³
        "used_satellite": selected_sat,
        "dates": dates,
        "sst_date": sst_date
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_chlorophyll(data: Location):
    """Predict chlorophyll-a concentration for a single location"""
    try:
        result = process_request(data.lat, data.lon, data.date)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Process multiple locations from CSV file"""
    try:
        df = pd.read_csv(file.file)
        
        # Validate required columns
        required_columns = ["lat", "lon"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        results = []
        for _, row in df.iterrows():
            try:
                result = process_request(
                    row["lat"], 
                    row["lon"], 
                    row.get("date", None)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing row {row.name}: {e}")
                results.append({
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "error": str(e)
                })
        
        return {"results": results, "total_processed": len(results)}
        
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    """API health check and information"""
    return {
        "message": "Chlorophyll-a Prediction API is running",
        "endpoints": {
            "/predict": "POST - Predict chlorophyll-a for single location",
            "/upload": "POST - Process CSV file with multiple locations",
            "/docs": "GET - API documentation"
        },
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)