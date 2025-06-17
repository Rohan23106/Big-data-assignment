# Hive schema and table command for task 2
CREATE EXTERNAL TABLE enriched_classifications_raw1 (
    video_id STRING,
    expected_label STRING,
    is_sludge BOOLEAN,
    confidence DOUBLE,
    layout_category STRING,
    summary STRING,
    text_features ARRAY<STRING>,
    text_tags ARRAY<STRING>,
    visual_features ARRAY<STRING>,
    visual_tags ARRAY<STRING>,
    recommendations ARRAY<STRING>
)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE
LOCATION 'gs://rohan2306/Metadata folder/';
----------------------------------------------------------------------------------

# Datawranglingcode for task 3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct

# Starting Spark session
spark = SparkSession.builder.appName("IntermediateWrangling").getOrCreate()

# Loading data from GCS
data_path = "gs://rohan2306/Metadata folder/enriched_classifications.jsonl"
df = spark.read.json(data_path)

# Printing schema
df.printSchema()
df.show(5)

# Dropping rows with missing important values
essential_columns = ["video_id", "expected_label", "is_sludge"]
df_clean = df.dropna(subset=essential_columns)

# Removing duplicates based on video_id
df_clean = df_clean.dropDuplicates(["video_id"])

# Showing counts by sludge classification
df_clean.groupBy("is_sludge").count().show()

# Showing counts by expected_label (e.g., TikTok / YouTube)
df_clean.groupBy("expected_label").count().show()

# Getting count of distinct video IDs
df_clean.select(countDistinct("video_id").alias("unique_video_count")).show()

# Saving cleaned output as JSON to GCS
output_path = "gs://rohan2306/output/cleaned_data_json"
df_clean.coalesce(1).write.mode("overwrite").json(output_path)

# Stopping Spark session
spark.stop()
-----------------------------------------------------------------------------------------------
# Hive Queries and table creation for task 3 after cleaning the dataset
CREATE EXTERNAL TABLE enriched_classifications_cleaned_csv1 (
    video_id STRING,
    expected_label STRING,
    is_sludge BOOLEAN,
    confidence DOUBLE,
    layout_category STRING,
    summary STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 'gs://rohan2306/Cleaned_data/'
TBLPROPERTIES ("skip.header.line.count"="1");

# Queries
Total number of videos:
SELECT COUNT(*) AS total_videos :-
FROM enriched_classifications_cleaned_csv1;

Count of Sludge vs Non-Sludge Videos :-
SELECT is_sludge, COUNT(*) AS video_count 
FROM enriched_classifications_cleaned_csv1
GROUP BY is_sludge;

Count by expected_label and is_sludge :-
SELECT expected_label, is_sludge, COUNT(*) AS count 
FROM enriched_classifications_cleaned_csv1
GROUP BY expected_label, is_sludge;

------------------------------------------------------------------------------
# pysparkMLlib1 codes for applying Machine learning, evaluation and visualization
# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc

# Starting Spark session
spark = SparkSession.builder.appName("SludgeClassifier").getOrCreate()

# Loading Cleaned CSV from GCS
data_path = "gs://rohan2306/Cleaned_data/Cleaneddata.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)

# Printing schema to verify data structure
df.printSchema()

# Converting is_sludge (boolean) to label (integer) for MLlib compatibility
df = df.withColumn("label", col("is_sludge").cast("integer"))

# Indexing the layout_category column (categorical -> numeric)
layout_indexer = StringIndexer(inputCol="layout_category", outputCol="layout_index", handleInvalid="keep")

# Assembling features into a single vector
feature_cols = ["confidence", "layout_index"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Defining the Random Forest classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)

# Creating the ML pipeline
pipeline = Pipeline(stages=[layout_indexer, assembler, rf])

# Splitting data into training and testing sets 
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Fitting the model to training data
model = pipeline.fit(train_data)

# Making predictions on test data
predictions = model.transform(test_data)

# Evaluating performance 
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)

# Evaluating additional classification metrics
acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
prec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="precisionByLabel")
rec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="recallByLabel")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

# Calculating evaluation scores
accuracy = acc_eval.evaluate(predictions)
precision = prec_eval.evaluate(predictions)
recall = rec_eval.evaluate(predictions)
f1 = f1_eval.evaluate(predictions)

# Printing performance results
print("\n--- Model Performance ---")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"ROC AUC:   {roc_auc:.3f}")

# Creating confusion matrix from predictions
cm_df = predictions.groupBy("label", "prediction").count().toPandas().pivot(index="label", columns="prediction", values="count").fillna(0)

# Defining local paths for saving plots
conf_matrix_path = "/tmp/confusion_matrix.png"
importance_path = "/tmp/feature_importance.png"
roc_path = "/tmp/roc_curve.png"

# Defining GCS bucket and upload destinations
bucket = "rohan2306"
gcs_conf_matrix = f"gs://{bucket}/visualizations/confusion_matrix.png"
gcs_importance = f"gs://{bucket}/visualizations/feature_importance.png"
gcs_roc_path = f"gs://{bucket}/visualizations/roc_curve.png"

# Plotting and saving confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True, fmt='g', cmap="Blues", xticklabels=["Non-Sludge", "Sludge"], yticklabels=["Non-Sludge", "Sludge"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.close()

# Extracting and plotting feature importances
rf_model = model.stages[-1]
importances = rf_model.featureImportances.toArray()
importance_df = pd.DataFrame(list(zip(feature_cols, importances)), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)

# Plotting and saving feature importance chart
plt.figure(figsize=(7,4))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(importance_path)
plt.close()

# Creating ROC Curve
roc_data = predictions.select("label", "probability").toPandas()
roc_data["probability"] = roc_data["probability"].apply(lambda x: float(x[1]))  # get probability of class 1

# Calculate FPR, TPR, AUC
fpr, tpr, _ = roc_curve(roc_data["label"], roc_data["probability"])
roc_auc_value = auc(fpr, tpr)

# Plotting and saving ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc_value:.3f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(roc_path)
plt.close()

# Uploading all visualizations to GCS
os.system(f"gsutil cp {conf_matrix_path} {gcs_conf_matrix}")
os.system(f"gsutil cp {importance_path} {gcs_importance}")
os.system(f"gsutil cp {roc_path} {gcs_roc_path}")

# Confirming upload
print("\n✅ Visualizations saved to GCS:")
print(f"{gcs_conf_matrix}")
print(f"{gcs_importance}")
print(f"{gcs_roc_path}")

------------------------------------------------------------------------------------------------
THE END
