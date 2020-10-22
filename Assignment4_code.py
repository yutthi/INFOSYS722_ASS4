#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('basics').getOrCreate()
from pyspark.sql.functions import mean,avg, isnan, when, count, col,udf
from pyspark.sql.functions import regexp_extract
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import BooleanType


# In[2]:


#read first data set
dt1 = spark.read.csv("enroll_districts_a.csv",inferSchema=True,header=True)
dt1.show()

#read second data set
dt2 = spark.read.csv("finance_districts_a.csv",inferSchema=True,header=True)
dt2.show()


# In[4]:


dt1.columns


# In[5]:


dt1.describe().show()


# In[4]:


#data description
dt2.describe().show()


# In[34]:


#data description
dt2['ENROLL','ENROLL','TOTALREV'].describe().show()


# In[35]:


dt1['DISTRICT','YEAR','A_A_A','G01-G08_A_A'].describe().show()


# In[36]:


dt2.columns


# In[37]:


dt1['DISTRICT','YEAR','A_A_A','KG_A_A','G01-G08_A_A','G09-G12_A_A','PK_A_A'].describe().show()


# In[3]:


dt1 = dt1.withColumnRenamed("G01-G08_A_A", "G01_G08_A_A")       .withColumnRenamed("G09-G12_A_A", "G09_G12_A_A")
dt1.show()
dt1.printSchema()


# In[7]:


dt2.printSchema()


# In[14]:


dt1_clean = dt1.dropna()
1-(dt1_clean.count()/dt1.count())


# In[15]:


dt2_clean = dt2.dropna()
1-(dt2_clean.count()/dt2.count())


# In[8]:


print((dt1.count(), len(dt1.columns)))


# In[16]:


print((dt2.count(), len(dt2.columns)))


# In[4]:




dt1.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dt1.columns]).show()


# In[23]:


dt2.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dt2.columns]).show()


# In[4]:


#DATA PREPARATION
dt1.createOrReplaceTempView('dt1')
dt1a = spark.sql("SELECT * FROM dt1 WHERE YEAR >= 2010 AND A_A_A>0")
dt1a.show()


# In[12]:


print((dt1a.count(), len(dt1a.columns)))


# In[5]:


dt1b = spark.sql("SELECT DISTRICT,YEAR,A_A_A,KG_A_A,G01_G08_A_A,G01_A_A,G02_A_A,G03_A_A,G04_A_A,G05_A_A,G06_A_A,G07_A_A,G08_A_A,G09_A_A,G10_A_A,G11_A_A,G12_A_A,G09_G12_A_A,PK_A_A FROM dt1 WHERE YEAR >= 2010 AND A_A_A>0")
dt1b.show()


# In[18]:


dt1b.select([count(when(isnan(c) | col(c).isNull(), c))/count(c).alias(c) for c in dt1b.columns]).show()


# In[6]:


dt1b.createOrReplaceTempView('dt1b')
dt1c = spark.sql("SELECT DISTRICT,YEAR,A_A_A,KG_A_A,G01_G08_A_A,G01_A_A,G02_A_A,G03_A_A,G04_A_A,G05_A_A,G06_A_A,G07_A_A,G08_A_A,G09_A_A,G10_A_A,G11_A_A,G12_A_A,G09_G12_A_A FROM dt1b ")
print((dt1c.count(), len(dt1c.columns)))


# In[7]:




def fill_with_mean(df, c): 
    mean_sales = df.select(mean(df[c])).collect()
    mean_sales_val = mean_sales[0][0]
    stats = df.na.fill(mean_sales_val, subset=[c])
    return stats
dt1d = dt1c

dt1d=fill_with_mean(dt1d, 'G01_A_A')
dt1d=fill_with_mean(dt1d, 'G02_A_A')
dt1d=fill_with_mean(dt1d, 'G03_A_A')
dt1d=fill_with_mean(dt1d, 'G04_A_A')
dt1d=fill_with_mean(dt1d, 'G05_A_A')
dt1d=fill_with_mean(dt1d, 'G06_A_A')
dt1d=fill_with_mean(dt1d, 'G07_A_A')
dt1d=fill_with_mean(dt1d, 'G08_A_A')
dt1d=fill_with_mean(dt1d, 'G09_A_A')
dt1d=fill_with_mean(dt1d, 'G10_A_A')
dt1d=fill_with_mean(dt1d, 'G11_A_A')
dt1d=fill_with_mean(dt1d, 'G12_A_A')
dt1d=fill_with_mean(dt1d, 'A_A_A')
dt1d=fill_with_mean(dt1d, 'KG_A_A')
dt1d=fill_with_mean(dt1d, 'G01_G08_A_A')
dt1d=fill_with_mean(dt1d, 'G09_G12_A_A')
dt1d.show()


# In[34]:


dt1d.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dt1d.columns]).show()


# In[8]:


dt2a = dt2
dt2a=fill_with_mean(dt2a, 'ENROLL')
dt2a=fill_with_mean(dt2a, 'TCURONON')
dt2a.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dt2a.columns]).show()


# In[9]:



def is_digit(value):
    if value:
        return value.isdigit()
    else:
        return False
    
is_digit_udf = udf(is_digit, BooleanType())

dt1d.createOrReplaceTempView('dt1d')
dt1e = dt1d.withColumn('DISTRICT', when(~is_digit_udf(dt1d['DISTRICT']), dt1d['DISTRICT']))
dt1e.show()


# In[10]:


dt1f = dt1e.withColumn('New_YEAR', when(is_digit_udf(dt1e['YEAR']), dt1e['YEAR']))


# In[30]:


print((dt1e.count(), len(dt1e.columns)))


# In[24]:


dt1e.describe().collect()


# In[11]:


dt1f = dt1e.withColumn('New_YEAR', regexp_extract(col('YEAR'), '.(-)*(\w+)', 2))
dt1f.show()


# In[26]:


#data description
dt1f['YEAR','A_A_A'].describe().show()


# In[12]:


dt1f.withColumn("high_prop", dt1f.G09_G12_A_A/dt1f.A_A_A).show()
dt1g = dt1f.withColumn("high_prop", dt1f.G09_G12_A_A/dt1f.A_A_A)


# In[13]:


dt1g.filter((dt1g['high_prop'] >0) & (dt1g['high_prop']<1)).show()
dt1h = dt1g.filter((dt1g['high_prop'] >0) & (dt1g['high_prop']<1))


# In[36]:


dt1h['YEAR','high_prop'].describe().show()


# In[14]:



def modify_values(r):
    if r <0.3 :
        return "LOW"
    else:
        if r <0.35:
            return "MID"
        else:
            return "HIGH"
    
ol_val = udf(modify_values, StringType())
dt1ha = dt1h.withColumn("Edu_group",ol_val(dt1h.high_prop))
dt1ha.show()


# In[15]:



df_a = dt1ha.filter(dt1ha['Edu_group'] == "HIGH")
df_b = dt1ha.filter(dt1ha['Edu_group'] == "MID")
df_c = dt1ha.filter(dt1ha['Edu_group'] == "LOW")

a_count = df_a.count()
b_count = df_b.count() 
c_count = df_c.count() 
high_ratio = c_count / a_count
mid_ratio = c_count / b_count

df_a_overampled = df_a.sample(withReplacement=True, fraction=high_ratio, seed=1)
df_b_overampled = df_b.sample(withReplacement=True, fraction=mid_ratio, seed=1)
df = df_c.unionAll(df_a_overampled)
dt1hb = df.unionAll(df_b_overampled)


# In[53]:


#before resample
dt1ha.groupBy('Edu_group').count().show()


# In[56]:


#after resample
dt1hb.groupBy('Edu_group').count().show()


# In[16]:


ta = dt1hb.alias('ta')
tb = dt2a.alias('tb')

full_dt = ta.join(tb, (ta.DISTRICT == tb.DISTRICT) & (ta.YEAR == tb.YEAR)).drop(tb.DISTRICT).drop(tb.YEAR) # Could also use 'left_outer'

full_dt.show()


# In[69]:


print((full_dt.count(), len(full_dt.columns)))


# In[17]:


full_dt1 = full_dt.withColumn("NEW_YEAR", full_dt["YEAR"].cast(StringType()))
full_dt1.show()


# In[18]:


full_dt2 = full_dt1.drop(full_dt1.NEW_YEAR).drop(full_dt1.DISTRICT).drop(full_dt1.high_prop)
full_dt2.printSchema()


# In[19]:




def modify_values1(r):
    if r =="LOW" :
        return 1
    else:
        if r =="MID":
            return 2
        else:
            return 3

ol_val1 = udf(modify_values1, IntegerType())
full_dt3 = full_dt2.withColumn("Edu_group1",ol_val1(full_dt2.Edu_group))
full_dt3.show()


# In[2]:


from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                                OneHotEncoder,StringIndexer)

State_indexer = StringIndexer(inputCol='STATE',outputCol='STATEIndex')

# Now we can one hot encode these numbers. This converts the various outputs into a single vector.
# This makes it easier to process when you have multiple classes.
State_encoder = OneHotEncoder(inputCol='STATEIndex',outputCol='STATEVec')
#Edu_group_indexer = StringIndexer(inputCol='Edu_group1',outputCol='Edu_groupIndex')
#Edu_group_encoder = OneHotEncoder(inputCol='Edu_groupIndex',outputCol='Edu_groupVec')


# In[3]:


assembler = VectorAssembler(inputCols=['YEAR',
 'A_A_A',
 'KG_A_A',
 'G01_G08_A_A',
 'G01_A_A',
 'G02_A_A',
 'G03_A_A',
 'G04_A_A',
 'G05_A_A',
 'G06_A_A',
 'G07_A_A',
 'G08_A_A',
 'G09_A_A',
 'G10_A_A',
 'G11_A_A',
 'G12_A_A',
 'G09_G12_A_A',
 'STATEVec',
 'ENROLL',
 'TOTALREV',
 'TFEDREV', 
 'TSTREV',
 'TLOCREV',
 'TOTALEXP', 
 'TCURINST',
 'TCURSSVC',
 'TCURONON',
 'TCAPOUT'],outputCol='features')


# In[4]:


from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[18]:




selector = ChiSqSelector(numTopFeatures=3, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="Edu_group1")

pipeline1 = Pipeline(stages=[State_indexer,State_encoder,
                           assembler,selector])

result = pipeline1.fit(full_dt3).transform(full_dt3)

print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
result.show()


# In[25]:


print((full_dt2.count(), len(full_dt2.columns)))


# In[7]:




dtc = DecisionTreeClassifier(labelCol='Edu_group1',featuresCol='features')
rfc = RandomForestClassifier(labelCol='Edu_group1',featuresCol='features')
#gbt = GBTClassifier(labelCol='Edu_group1',featuresCol='features')

pipeline1 = Pipeline(stages=[State_indexer,State_encoder,
                           assembler,dtc])
pipeline2 = Pipeline(stages=[State_indexer,State_encoder,
                           assembler,rfc])
#pipeline3 = Pipeline(stages=[State_indexer,State_encoder,assembler,gbt])
                           
train_data,test_data = full_dt3.randomSplit([0.7,0.3])

dtc_model = pipeline1.fit(train_data)
rfc_model = pipeline2.fit(train_data)
#gbt_model = pipeline3.fit(train_data)

dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
#gbt_predictions = gbt_model.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Select (prediction, true label) and compute test error. 
acc_evaluator = MulticlassClassificationEvaluator(labelCol="Edu_group1", predictionCol="prediction", metricName="accuracy")
dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
#gbt_acc = acc_evaluator.evaluate(gbt_predictions)
print("Here are the results!")
print('-'*40)
print('A single decision tree has an accuracy of: {0:2.2f}%'.format(dtc_acc*100))
print('-'*40)
print('A random forest ensemble has an accuracy of: {0:2.2f}%'.format(rfc_acc*100))


# In[26]:


full_dt3.write.save("data1.parquet")


# In[6]:


full_dt3 = spark.read.load("data1.parquet")


# In[35]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

train_data,test_data = full_dt3.randomSplit([0.7,0.3])
train_data = train_data.dropna()
test_data = test_data.dropna()
lr = LinearRegression(featuresCol='features', labelCol='Edu_group1', predictionCol='prediction')

pipeline2 = Pipeline(stages=[State_indexer,State_encoder,assembler,lr])

lr_model = pipeline2.fit(train_data)
# Transform test data. 
#test_results = lr_model.evaluate(test_data)
lr_predictions = lr_model.transform(test_data)
lr_acc = acc_evaluator.evaluate(lr_predictions)
print('-'*40)
print('A Linear Regression Model has an accuracy of: {0:2.2f}%'.format(lr_acc*100))

# Interesting results! This shows the difference between the predicted value and the test data.
#test_results.residuals.show()

# Let's get some evaluation metrics (as discussed in the previous linear regression notebook).
#print("RSME: {}".format(test_results.rootMeanSquaredError))


# In[10]:


import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType

#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = dtc_predictions.select(['prediction','Edu_group1']).withColumn('label', F.col('Edu_group1').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())


# In[11]:


preds_and_labels = rfc_predictions.select(['prediction','Edu_group1']).withColumn('label', F.col('Edu_group1').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())


# In[11]:


[stage.coefficients for stage in lr_model.stages if hasattr(stage, "coefficients")]


# In[12]:


[stage.coefficients for stage in lr_model.stages if hasattr(stage, "intercept")]


# In[18]:


print((train_data.count(), len(train_data.columns)))
print((test_data.count(), len(test_data.columns)))


# In[14]:


full_dt4 = full_dt3.withColumn("FEDREV_prop", full_dt3.TFEDREV/full_dt3.TOTALREV)
full_dt4 = full_dt4.withColumn("TLOCREV_prop", full_dt4.TLOCREV/full_dt4.TOTALREV)
full_dt4.show()


# In[15]:


assembler1 = VectorAssembler(inputCols=['YEAR',
 'A_A_A',
 'KG_A_A',
 'G01_G08_A_A',
 'G01_A_A',
 'G02_A_A',
 'G03_A_A',
 'G04_A_A',
 'G05_A_A',
 'G06_A_A',
 'G07_A_A',
 'G08_A_A',
 'G09_A_A',
 'G10_A_A',
 'G11_A_A',
 'G12_A_A',
 'G09_G12_A_A',
 'STATEVec',
 'ENROLL',
 'TOTALREV',
 'TFEDREV', 
 'TSTREV',
 'TLOCREV',
 'TOTALEXP', 
 'TCURINST',
 'TCURSSVC',
 'TCURONON',
 'TCAPOUT'],outputCol='features')

dtc = DecisionTreeClassifier(labelCol='Edu_group1',featuresCol='features')
rfc = RandomForestClassifier(labelCol='Edu_group1',featuresCol='features')
#gbt = GBTClassifier(labelCol='Edu_group1',featuresCol='features')

pipeline3 = Pipeline(stages=[State_indexer,State_encoder,
                           assembler1,dtc])
pipeline4 = Pipeline(stages=[State_indexer,State_encoder,
                           assembler1,rfc])
#pipeline3 = Pipeline(stages=[State_indexer,State_encoder,assembler,gbt])
                           
train_data,test_data = full_dt4.randomSplit([0.7,0.3])

dtc_model2 = pipeline3.fit(train_data)
rfc_model2 = pipeline4.fit(train_data)
#gbt_model = pipeline3.fit(train_data)

dtc_predictions2 = dtc_model2.transform(test_data)
rfc_predictions2 = rfc_model2.transform(test_data)

acc_evaluator = MulticlassClassificationEvaluator(labelCol="Edu_group1", predictionCol="prediction", metricName="accuracy")
dtc_acc2 = acc_evaluator.evaluate(dtc_predictions2)
rfc_acc2 = acc_evaluator.evaluate(rfc_predictions2)
#gbt_acc = acc_evaluator.evaluate(gbt_predictions)
print("Here are the results!")
print('-'*40)
print('A single decision tree has an accuracy of: {0:2.2f}%'.format(dtc_acc2*100))
print('-'*40)
print('A random forest ensemble has an accuracy of: {0:2.2f}%'.format(rfc_acc2*100))


# In[16]:


preds_and_labels = dtc_predictions2.select(['prediction','Edu_group1']).withColumn('label', F.col('Edu_group1').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())


# In[17]:


preds_and_labels = rfc_predictions2.select(['prediction','Edu_group1']).withColumn('label', F.col('Edu_group1').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','label'])

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print(metrics.confusionMatrix().toArray())


# In[ ]:




