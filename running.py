from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("5km Runners Analysis") \
    .getOrCreate()

df = spark.read.csv(r"/Users/admin/Projects/BigData/expanded_dataset.csv", header=True, inferSchema=True)

df = df.drop(df.columns[0])

df = df.withColumn('Age', regexp_extract(col('Age'), '(\d+)', 0).cast('int'))

df = df.withColumn('Gender', when(col('Gender') == 'М', 1).otherwise(0))

def time_to_hours(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        parts.insert(0, '0')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds / 3600

time_to_hours_udf = udf(time_to_hours, DoubleType())
df = df.withColumn('Time', time_to_hours_udf(col('Time')))

pandas_df = df.toPandas()

plt.figure(figsize=(10, 6))
sns.countplot(x='Age', data=pandas_df)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 8))
gender_counts = pandas_df['Gender'].value_counts()
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightpink'])
plt.title('Gender Distribution')
plt.show()

assembler = VectorAssembler(inputCols=['Age', 'Gender'], outputCol='features')
data = assembler.transform(df)

df = df.repartition(10)
df.cache()
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

gbt = GBTRegressor(featuresCol='features', labelCol='Time', maxIter=50)

gbt_model = gbt.fit(train_data)

predictions = gbt_model.transform(test_data)

evaluator = RegressionEvaluator(labelCol='Time', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

# Output results
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
print(f"R2 score on test data = {r2}")

predictions_shuffled = predictions.withColumn("random", F.rand()).orderBy("random").drop("random")
predictions_shuffled.select("Age", "Gender", "Time", "prediction").show(400)
