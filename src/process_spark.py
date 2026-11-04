import os
import sys

# Make Spark use the Python from our .venv (avoids a common Windows error)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Windows only: tell Spark where the Hadoop helper files are
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin" + os.pathsep + os.environ["PATH"]

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from config import HORIZONS, PROCESSED_DIR, RAW_DIR

NUMERIC_COLS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "us_aqi",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation", "cloud_cover",
]
PM_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]   # hours
ROLL_WINDOWS = [6, 24, 168]                    # hours


def main():
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("AirQualityProcessing")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ---------- 1. LOAD ----------
    df = spark.read.parquet(RAW_DIR.as_posix())
    df = df.withColumn("time", F.col("time").cast("timestamp"))
    print(f"Loaded {df.count():,} rows")

    # ---------- 2. CLEAN ----------
    df = df.dropDuplicates(["city", "time"])

    w_city = Window.partitionBy("city").orderBy("time")
    w_ffill = w_city.rowsBetween(Window.unboundedPreceding, 0)

    # Fill any gaps with the last known value (safety net for future data)
    for c in NUMERIC_COLS:
        df = df.withColumn(c, F.last(c, ignorenulls=True).over(w_ffill))

    # Concentrations can't be negative
    df = df.withColumn("pm2_5", F.greatest(F.col("pm2_5"), F.lit(0.0)))
    df = df.dropna(subset=["pm2_5"])

    # ---------- 3. FEATURES ----------
    # 3a. Lag features: PM2.5 N hours ago
    for h in PM_LAGS:
        df = df.withColumn(f"pm2_5_lag_{h}h", F.lag("pm2_5", h).over(w_city))

    # 3b. Rolling statistics (current hour and the past only)
    for n in ROLL_WINDOWS:
        w = w_city.rowsBetween(-(n - 1), 0)
        df = df.withColumn(f"pm2_5_roll_mean_{n}h", F.avg("pm2_5").over(w))
    w24 = w_city.rowsBetween(-23, 0)
    w168 = w_city.rowsBetween(-167, 0)
    df = (
        df.withColumn("pm2_5_roll_std_24h", F.stddev("pm2_5").over(w24))
        .withColumn("pm2_5_roll_max_24h", F.max("pm2_5").over(w24))
        .withColumn("pm2_5_roll_std_168h", F.stddev("pm2_5").over(w168))
    )

    # 3c. Weather features (rain washes pollution out, wind disperses it)
    df = (
        df.withColumn("wind_roll_mean_24h", F.avg("wind_speed_10m").over(w24))
        .withColumn("humidity_roll_mean_24h", F.avg("relative_humidity_2m").over(w24))
        .withColumn("temp_roll_mean_24h", F.avg("temperature_2m").over(w24))
        .withColumn("precip_sum_24h", F.sum("precipitation").over(w24))
        .withColumn(
            "pressure_change_24h",
            F.col("surface_pressure") - F.lag("surface_pressure", 24).over(w_city),
        )
    )

    # 3d. Calendar features (cyclical encoding so hour 23 is "near" hour 0)
    two_pi = 2 * 3.141592653589793
    df = (
        df.withColumn("hour", F.hour("time"))
        .withColumn("day_of_week", F.dayofweek("time"))
        .withColumn("month", F.month("time"))
        .withColumn("day_of_year", F.dayofyear("time"))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0))
        .withColumn("hour_sin", F.sin(F.col("hour") * two_pi / 24))
        .withColumn("hour_cos", F.cos(F.col("hour") * two_pi / 24))
        .withColumn("doy_sin", F.sin(F.col("day_of_year") * two_pi / 365.25))
        .withColumn("doy_cos", F.cos(F.col("day_of_year") * two_pi / 365.25))
        .withColumn("wind_dir_sin", F.sin(F.radians("wind_direction_10m")))
        .withColumn("wind_dir_cos", F.cos(F.radians("wind_direction_10m")))
    )

    # ---------- 4. TARGETS: PM2.5 in the future ----------
    for h in HORIZONS:
        df = df.withColumn(f"target_{h}h", F.lead("pm2_5", h).over(w_city))

    # Drop the first 7 days of each city (not enough history for lag_168h)
    df = df.dropna(subset=["pm2_5_lag_168h", "pm2_5_roll_std_168h", "pressure_change_24h"])

    # ---------- 5. SAVE ----------
    # (Last 72 hours of each city have no targets yet; we keep them
    #  because they are exactly the rows we forecast FROM.)
    pdf = df.orderBy("city", "time").toPandas()
    out_file = PROCESSED_DIR / "features.parquet"
    pdf.to_parquet(out_file, index=False)
    print(f"Saved {len(pdf):,} rows x {pdf.shape[1]} columns -> {out_file}")

    # Small summary table for the dashboard later (a few KB)
    monthly = (
        df.withColumn("year_month", F.date_format("time", "yyyy-MM"))
        .groupBy("city", "year_month")
        .agg(F.round(F.avg("pm2_5"), 1).alias("avg_pm2_5"),
             F.round(F.max("pm2_5"), 1).alias("max_pm2_5"))
        .orderBy("city", "year_month")
        .toPandas()
    )
    monthly.to_csv(PROCESSED_DIR / "city_monthly_summary.csv", index=False)
    print("Saved monthly summary")

    spark.stop()


if __name__ == "__main__":
    main()