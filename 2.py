from kafka import KafkaConsumer, KafkaProducer
import datetime as dt
import json
import pandas as pd
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost.spark import SparkXGBClassifierModel
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit, col
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


COLUMNS_LIST = ["StartTime", "Dur", "TotPkts", "TotBytes", "SrcBytes", "Proto_tcp", "Proto_udp", "Dir_one",
           "sTosone", "Proto_others", "Dir_others", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", 
           "Dport", "State", "sTos", "dTos", "Label"]

def doingpotty(df):
    model_save_path = "Model/pyspark"
    loaded_model = SparkXGBClassifierModel.load(model_save_path)
    feature_cols = ['Dur', 'TotPkts', 'TotBytes', 'SrcBytes', 'Proto_tcp', 'Proto_udp', 'Dir_one', 'sTosone', 'Proto_others', 'Dir_others']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_new = assembler.transform(df)
    selected_cols = ["StartTime", "features", "Label"]
    df_test = df_new.select(selected_cols)
    df_test = df_test.withColumn("Label", col("Label").cast('int'))

    predictions = loaded_model.transform(df_test)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy: {accuracy}")
    predictions = predictions.select(["StartTime", "prediction"])
    predictions = predictions.withColumn("prediction", col("prediction").cast('int'))
    df_finalboob = df.join(predictions, on='StartTime')
    final_cols = ["StartTime", "Dur", "TotPkts", "TotBytes", "SrcBytes", "Proto_tcp", "Proto_udp", "Dir_one",
           "sTosone", "Proto_others", "Dir_others", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", 
           "Dport", "State", "sTos", "dTos", "prediction", "Label"]
    df_finalboob = df_finalboob.select(final_cols)
    return df_finalboob

consumer = KafkaConsumer('logsprocessed',
                         group_id='test-consumer-group',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                         auto_offset_reset='earliest',
                         enable_auto_commit=True)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
print(f'Initialized Kafka producer at {dt.datetime.utcnow()}')

"""----------------------------------------------------------------------------------------------------------------"""

for message in consumer:
    rows = [Row(**{key: value[str(idx)] for key, value in message.value.items()}) for idx in range(len(message.value['StartTime']))]
    spark = SparkSession.builder.appName("example").getOrCreate()
    df = spark.createDataFrame(rows)
    combined_df = doingpotty(df)
    row = combined_df.first()
    row_dict = row.asDict()
    data = json.dumps(row_dict, default=str).encode('utf-8')
    print(data)
    print("-" * 75)
    producer.send(topic="logslabelled", value=data)