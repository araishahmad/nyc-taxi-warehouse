# NYC Taxi Data Warehouse

A end-to-end big data pipeline built with the Hadoop ecosystem that ingests, processes, and analyzes NYC TLC taxi trip records for fare prediction, hotspot mapping, and driver performance analytics.

---

## Project Overview

This project builds a fully functional data warehouse using real NYC government taxi data — over 3 million trip records from January 2023 alone. The pipeline ingests raw Parquet files, cleans and transforms them using distributed computing, exposes them through a SQL query layer, trains a machine learning model for fare prediction, and visualizes insights through interactive dashboards.

---

## Tech Stack

| Tool | Role |
|------|------|
| **Apache Hadoop (HDFS)** | Distributed storage for raw and processed trip data |
| **Apache Spark (PySpark)** | Data cleaning, feature engineering, ML model training |
| **Apache Hive** | SQL query layer on top of HDFS |
| **Apache Superset** | Interactive dashboards and visualizations |
| **Docker Compose** | Full stack orchestration |
| **Python** | Pipeline scripting and ML |

---

## Architecture

```
NYC TLC Website (Raw Parquet files)
        │
        ▼
HDFS /taxi/raw/          ← Distributed storage
        │
        ▼
Apache Spark             ← Clean, transform, feature engineer
        │
        ▼
HDFS /taxi/processed/    ← Clean Parquet, partitioned by hour_of_day
        │
        ▼
Apache Hive              ← SQL interface on top of HDFS
        │
        ▼
Apache Superset          ← Dashboards and visualizations
```

---

## Project Goals

- **Fare Prediction** — Train an ML model to predict taxi fares from trip features
- **Hotspot Mapping** — Identify top pickup locations across NYC boroughs
- **Driver Performance** — Analyze trip patterns by hour, zone, and borough

---

## Key Results

| Metric | Value |
|--------|-------|
| Raw trip records (Jan 2023) | 3,066,766 |
| Clean records after processing | 2,879,928 |
| Bad rows removed | ~187,000 |
| Fare prediction R² | 0.9317 (93%) |
| Fare prediction RMSE | $4.41 |
| Top pickup location | JFK Airport — 151,653 trips |
| Busiest hour | 6pm — 203,349 trips |
| Highest avg fare hour | 4am — $26.38 |
| Manhattan share of all trips | 88.8% |

---

## What the ML Model Learned

The Linear Regression model trained on 2.3M trips independently discovered NYC taxi meter rates from raw data alone — without being told the actual rates:

```
Predicted fare = $3.60 + (distance × $3.02) + (duration × $0.32)
```

Actual NYC published meter rates:
- Base fare: $3.00
- Per mile: $2.50
- Per minute: $0.50

---

## Project Structure

```
nyc-taxi-warehouse/
├── docker-compose.yml          # Full stack orchestration
├── hadoop.env                  # Hadoop configuration
├── clean_taxi_data.py          # Spark cleaning and feature engineering job
├── zone_analysis.py            # Spark zone join and borough analysis
├── fare_prediction.py          # Spark MLlib fare prediction model
├── export_for_superset.py      # Export aggregated CSVs for dashboards
├── data/
│   ├── yellow_tripdata_2023-01.parquet   # Raw NYC TLC trip records
│   └── taxi_zone_lookup.csv             # Zone ID to neighborhood mapping
└── exports/
    ├── trips_by_hour/          # Aggregated trips per hour
    ├── top_pickup_zones/       # Top 50 pickup zones with names
    └── borough_summary/        # Trip summary by borough
```

---

## Setup & Installation

### Prerequisites
- Docker Desktop with WSL2 enabled (Windows) or Docker Engine (Linux/Mac)
- 16GB RAM recommended (8GB minimum)
- 50GB free disk space
- Python 3.8+

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/nyc-taxi-warehouse.git
cd nyc-taxi-warehouse
```

### 2. Start the stack
```bash
docker compose up -d
```

Wait 60 seconds for all containers to initialize.

### 3. Initialize Hive metastore schema
```bash
docker exec -it hive-metastore /opt/hive/bin/schematool -upgradeSchemaFrom 2.3.0 -dbType postgres
```

### 4. Verify all containers are running
```bash
docker compose ps
```

All containers should show `Up`.

---

## Running the Pipeline

### Step 1 — Download the data
Download from the [NYC TLC website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):
- `yellow_tripdata_2023-01.parquet`
- `taxi_zone_lookup.csv`

Place both in the `data/` folder.

### Step 2 — Load data into HDFS
```bash
docker exec namenode hdfs dfs -mkdir -p /taxi/raw/2023/01
docker exec namenode hdfs dfs -mkdir -p /taxi/lookup
docker cp data/yellow_tripdata_2023-01.parquet namenode:/tmp/
docker cp data/taxi_zone_lookup.csv namenode:/tmp/
docker exec namenode hdfs dfs -put /tmp/yellow_tripdata_2023-01.parquet /taxi/raw/2023/01/
docker exec namenode hdfs dfs -put /tmp/taxi_zone_lookup.csv /taxi/lookup/
```

### Step 3 — Run the Spark cleaning job
```bash
docker cp clean_taxi_data.py spark-master:/tmp/
docker exec -e PYSPARK_PYTHON=python3 spark-master /spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /tmp/clean_taxi_data.py
```

### Step 4 — Create Hive tables
```bash
docker exec -it hive-metastore /opt/hive/bin/hive
```

```sql
CREATE DATABASE IF NOT EXISTS taxi_warehouse;
USE taxi_warehouse;

CREATE EXTERNAL TABLE IF NOT EXISTS taxi_warehouse.taxi_trips (
    tpep_pickup_datetime    TIMESTAMP,
    tpep_dropoff_datetime   TIMESTAMP,
    passenger_count         DOUBLE,
    trip_distance           DOUBLE,
    PULocationID            INT,
    DOLocationID            INT,
    fare_amount             DOUBLE,
    tip_amount              DOUBLE,
    total_amount            DOUBLE,
    trip_duration_mins      DOUBLE,
    speed_mph               DOUBLE
)
PARTITIONED BY (hour_of_day INT)
STORED AS PARQUET
LOCATION 'hdfs://namenode:9000/taxi/processed/2023/01/';

MSCK REPAIR TABLE taxi_warehouse.taxi_trips;
```

### Step 5 — Train the fare prediction model
```bash
docker cp fare_prediction.py spark-master:/tmp/
docker exec -e PYSPARK_PYTHON=python3 spark-master /spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /tmp/fare_prediction.py
```

### Step 6 — Set up Superset dashboards
```bash
docker exec -it superset superset db upgrade
docker exec -it superset superset fab create-admin \
  --username admin --firstname Admin --lastname User \
  --email admin@example.com --password admin123
docker exec -it superset superset init
```

Open `http://localhost:8088` and login with `admin` / `admin123`.

---

## Web UIs

| Service | URL | Description |
|---------|-----|-------------|
| HDFS NameNode | http://localhost:9870 | Browse HDFS filesystem |
| Spark Master | http://localhost:8080 | Monitor Spark jobs |
| Spark Worker | http://localhost:8081 | Worker stats |
| Superset | http://localhost:8088 | Analytics dashboards |

---

## Dashboards

The Superset dashboard includes four charts built from real trip data:

- **Trips by Hour of Day** — 24-hour trip volume showing peak at 6pm
- **Average Fare by Hour** — Fare patterns showing 4am premium pricing
- **Top Pickup Zones** — JFK Airport, Upper East Side, Midtown leading
- **Trips by Borough** — Manhattan dominates at 88.8% of all trips

---

## Data Source

NYC Taxi & Limousine Commission (TLC) Trip Record Data  
[https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Yellow taxi trip records, January 2023. Published under NYC Open Data.

---

## License

MIT License — free to use, modify, and distribute.