# Databricks notebook source
# MAGIC %md  #SF crime data analysis and modeling

# COMMAND ----------

from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import os
import datetime
from pyspark.sql.functions import year, month, dayofmonth
os.environ["PYSPARK_PYTHON"] = "python3"
%matplotlib inline

# COMMAND ----------

#download the data from official website
import urllib.request
urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/myxxxx.csv")
dbutils.fs.mv("file:/tmp/myxxxx.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))

# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"

# COMMAND ----------

#get the dataframe 
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

from pyspark.sql.functions import to_date, to_timestamp, hour
df_opt1 = df_opt1.withColumn('Date', to_date(df_opt1.OccurredOn, "MM/dd/yy"))
df_opt1 = df_opt1.withColumn('Time', to_timestamp(df_opt1.OccurredOn, "MM/dd/yy HH:mm"))
df_opt1 = df_opt1.withColumn('Hour', hour(df_opt1['Time']))
df_opt1 = df_opt1.withColumn("DayOfWeek", date_format(df_opt1.Date, "EEEE"))

# COMMAND ----------

# MAGIC %md ##1.counts the number of crimes for different category.

# COMMAND ----------

crimeCategory = spark.sql("SELECT  category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(crimeCategory)

# COMMAND ----------

#visualize the result 
crimes_pd_df = crimeCategory.toPandas()
plt.figure(figsize=(20, 10))
chart = sns.barplot(x = 'category', y = 'Count', palette= 'mako',data = crimes_pd_df )
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.show()

# COMMAND ----------

# MAGIC %md ##2.counts the number of crimes for different district.

# COMMAND ----------

#Counts the number of crimes for different district
crimeDistrict = spark.sql("SELECT  PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY PdDistrict ORDER BY Count DESC")
display(crimeDistrict)

# COMMAND ----------

#visualize the result 
criDistrict_pd_df = crimeDistrict.toPandas()
plt.figure(figsize=(10, 5))
chart = sns.barplot(x = 'PdDistrict', y = 'Count', palette= 'mako',data = criDistrict_pd_df )
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.show()

# COMMAND ----------

# MAGIC %md ##3.Count the number of crimes each "Sunday" at "SF downtown".

# COMMAND ----------

#Count the number of crimes each "Sunday" at "SF downtown".
sundayCrime = spark.sql("""SELECT Date, COUNT(*) AS Count FROM sf_crime WHERE DayOfWeek = 'Sunday'
                          AND X > -122.4313 AND X < -122.4213 AND Y > 37.7540 AND Y < 37.7740 
                          GROUP BY Date ORDER BY Date""")

# COMMAND ----------

#visualize the result 
display(sundayCrime)

# COMMAND ----------

# MAGIC %md ## 4. Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?

# COMMAND ----------

#count the number of crime in each month of 2015, 2016, 2017, 2018.
crimeMonthly = spark.sql("""
                    WITH cm AS (
                    SELECT Date, substring(Date,7) As Year, substring(Date,1,2) As Month 
                    FROM sf_crime
                    )
          
                    SELECT Year, Month, COUNT(*) AS Count 
                    FROM cm 
                    WHERE Year BETWEEN 2015 AND 2018 
                    GROUP BY Year, Month 
                    ORDER BY Year, Month""")
display(crimeMonthly)

# COMMAND ----------

# MAGIC %md 
# MAGIC 1. Through the graph, we can see that the number of crimes decreased sharply from 2017 to 2018, and in 2015-2017, the number of crimes are basically flat. 
# MAGIC 2. The reduction of the number of crime cases will bring economic development. For companies, they can choose to open up more offline stores in SF.

# COMMAND ----------

# MAGIC %md ## 5. Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF.

# COMMAND ----------

from pyspark.sql.types import FloatType
from pyspark.sql.functions import col, udf
hour_func = udf (lambda x: float(x[:2]), FloatType())
df = df_opt1.withColumn('hour', hour_func(col('Time')))
crimeHourly = df.filter((df.Date == '12/15/2015') | (df.Date == '12/15/2016') | (df.Date == '12/15/2017'))\
       .groupBy('hour').count()
display(crimeHourly)

# COMMAND ----------

# MAGIC %md Through this graph we can see that the peaks of crime happened during the lunch time, dinner time and midnight. I would suggest the tourists pay attention to personal safety during those timeslots. 

# COMMAND ----------

# MAGIC %md ##6
# MAGIC    ##(1) Step1: Find out the top-3 danger disrict
# MAGIC    ##(2) Step2: find out the crime event w.r.t category and time (hour)   from the result of step 1
# MAGIC    ##(3) give your advice to distribute the police based on your analysis results.

# COMMAND ----------

#count the crime of each district
crimeTopDistict = df_opt1.groupBy('PdDistrict').count().orderBy('count', ascending=False).limit(3)
display(crimeTopDistict)

# COMMAND ----------

crimeCategoryTime = df.filter((df.PdDistrict == 'SOUTHERN') | (df.PdDistrict == 'MISSION') | (df.PdDistrict == 'NORTHERN'))\
            .groupBy('category', 'hour').count().orderBy('hour')
display(crimeCategoryTime)

# COMMAND ----------

# MAGIC %md From the chart we can see that the majority of crimes are larceny and theft. The peak hours of crimes cases are dinner time, lunch time and midnight. For police, more human resources can be deployed during these timeslots.

# COMMAND ----------

# MAGIC %md ##7. For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

crimeResolution = spark.sql("""
                   WITH cr As (
                      SELECT Category, Resolution,
                             CASE WHEN Resolution In ('NONE') then 0
                             ELSE 1 end As isResolution
                      FROM sf_crime  
                   )
                   
                   SELECT Category, isResolution, Count(*) As Count
                   FROM cr
                   GROUP BY Category, isResolution
                   ORDER BY Category, isResolution
                   """)
display(crimeResolution)

# COMMAND ----------

#Find the percentage of resolution for different crime category 
crimeResPercent = spark.sql("""
                   WITH crp As (
                      SELECT Category, Resolution,
                             CASE WHEN Resolution In ('NONE') then 0
                             ELSE 1 end As isResolution
                      FROM sf_crime  
                   )
                   
                   SELECT Category, Count(*) As Count,
                          count( CASE WHEN isResolution = 1 then isResolution end) As CountRes,
                          count( CASE WHEN isResolution = 1 then isResolution end) * 100 / Count(*) As percent
                   FROM crp
                   GROUP BY Category
                   ORDER BY percent 
                   """)
display(crimeResPercent)

# COMMAND ----------

# MAGIC %md The percentage of resolution for recovered vehicle, vehicle theft and larceny/theft are the three lowest. Larceny/theft is the most happened crime cases in SF, at the same time, it's resolution rate is low. Police can pay more attention on solving the theft-related cases, also, people in SF can increase vigilance regarding theft, such as improve their locker's security for their houses, pay more attention to their personal belongings in public, and so on. 
