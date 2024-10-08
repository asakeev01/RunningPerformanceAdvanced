# RunningPerformanceAdvanced
## Main project - https://github.com/asakeev01/RunningPerformance
## Updates for Handling Large Datasets and Using GBTRegressor

This section highlights the updates made to the project to accommodate a larger dataset (161,398 rows) and the switch from `KNeighborsRegressor` to `GBTRegressor` for machine learning.

### Machine Learning with Spark
For handling larger datasets, we switched from `KNeighborsRegressor` to `GBTRegressor` using Apache Spark for distributed computing. The following updates were made:

- **Model**: `GBTRegressor` replaces KNN
  - `maxIter=100`
  - `maxDepth=5`
- **Libraries**: Apache Spark MLlib for machine learning, PySpark for data processing
- **Evaluation Metrics**: Root Mean Squared Error (RMSE), R² Score

### Example Code Changes
Key code changes involved switching to Spark and implementing `GBTRegressor`:

```python
from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GBTRegressor Example") \
    .getOrCreate()

# Data preparation
assembler = VectorAssembler(inputCols=['Age', 'Gender'], outputCol='features')
gbt = GBTRegressor(featuresCol='features', labelCol='Time', predictionCol='prediction')

# Create pipeline and fit model
pipeline = Pipeline(stages=[assembler, gbt])
model = pipeline.fit(df)

# Predictions
predictions = model.transform(df)
```

### Visualization Updates
The data is converted from a Spark DataFrame to Pandas for visualization purposes:

```python
predictions_pd = predictions.select('Time', 'prediction').toPandas()
