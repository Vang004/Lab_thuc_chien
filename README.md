# Lab_thuc_chien

# 1. Cài đặt Java 11
!apt-get update -qq
!apt-get install openjdk-11-jdk-headless -qq > /dev/null

# 2. ÉP CÀI ĐẶT PYSPARK 3.5.1 (Bản ổn định nhất, không bị lỗi tương thích)
# Bỏ qua bản 4.1.1 đang bị lỗi của Colab
!pip install -q pyspark==3.5.1

# 3. Tải dữ liệu thực hành
!wget -q https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv -O retail_data.csv

# 4. Khai báo biến môi trường cho Java
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# 5. Khởi tạo Spark Session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

print("⏳ Đang khởi động hệ thống core park 3.5.1...")
spark = SparkSession.builder \
    .appName("RealWorld_DataProcessing") \
    .getOrCreate()

print("✅ HỆ THỐNG SPARK ĐÃ KẾT NỐI THÀNH CÔNG.")

# Đọc file CSV (Dùng inferSchema để Spark tự đoán kiểu dữ liệu)
df_raw = spark.read.csv("retail_data.csv", header=True, inferSchema=True)

# Khám phá cơ bản
print(f"Tổng số dòng ban đầu: {df_raw.count()}")
df_raw.printSchema()
df_raw.show(5)

# Thống kê mô tả để tìm lỗi
df_raw.describe("Quantity", "UnitPrice").show()

# BÀI TẬP 1: DỌN DẸP RÁC
df_clean = df_raw.filter(
    col("CustomerID").isNotNull() &
    (col("Quantity") > 0) &
    (col("UnitPrice") > 0)
)

# Kiểm tra kết quả
print(f"Số dòng sau khi làm sạch: {df_clean.count():,}")
df_clean.describe("Quantity", "UnitPrice").show()

# Cú pháp chuyển String -> Timestamp với định dạng cụ thể (M/d/yyyy H:mm)
# Giả sử bạn đã làm xong df_clean ở trên, lấy df_clean để làm tiếp.
# (Ở đây tạo biến tạm để code không bị lỗi nếu sinh viên chưa làm Bài 1)

df_parsed = df_raw.withColumn(
    "InvoiceDate",
    to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm")
)
df_parsed.select("InvoiceNo", "InvoiceDate").show(5)

# BÀI TẬP 2: BIẾN ĐỔI & TÍNH DOANH THU

# 1. Chuyển InvoiceDate từ String → Timestamp
df_clean = df_clean.withColumn(
    "InvoiceDate",
    to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm")
)

# 2. Tạo các cột mới
df_transformed = df_clean \
    .withColumn("TotalAmount", col("Quantity") * col("UnitPrice")) \
    .withColumn("InvoiceYear", year(col("InvoiceDate"))) \
    .withColumn("InvoiceMonth", month(col("InvoiceDate")))

# Kiểm tra
df_transformed.select("InvoiceNo", "InvoiceDate", "TotalAmount", "InvoiceYear", "InvoiceMonth").show(5)
print(f"Số dòng sau khi transform: {df_transformed.count():,}")

# BÀI TẬP 3: TOP 5 QUỐC GIA
top5_countries = df_transformed.groupBy("Country") \
    .agg(sum("TotalAmount").alias("Total_Revenue")) \
    .orderBy(desc("Total_Revenue")) \
    .limit(5)

top5_countries.show(truncate=False)

# BÀI TẬP 4: LƯU PARQUET PHÂN VÙNG
df_transformed.write \
    .mode("overwrite") \
    .partitionBy("Country") \
    .parquet("gold_sales_data")

print("✅ ĐÃ LƯU THÀNH CÔNG VÀO THƯ MỤC: gold_sales_data/")
print("Cấu trúc phân vùng theo Country đã được tạo!")

# CÀI ĐẶT PLOTLY (chạy 1 lần)
!pip install -q plotly

import plotly.express as px

# 1. DOANH THU THEO THÁNG
monthly_revenue = df_transformed.groupBy("InvoiceYear", "InvoiceMonth") \
    .agg(sum("TotalAmount").alias("TotalRevenue")) \
    .orderBy("InvoiceYear", "InvoiceMonth")

pdf_month = monthly_revenue.toPandas()
pdf_month["YearMonth"] = pdf_month["InvoiceYear"].astype(str) + "-" + \
                         pdf_month["InvoiceMonth"].astype(str).str.zfill(2)

fig1 = px.line(pdf_month, x="YearMonth", y="TotalRevenue",
               title="📈 Doanh thu theo Tháng (2010-2011)",
               markers=True,
               labels={"TotalRevenue": "Doanh thu (£)", "YearMonth": "Thời gian"})
fig1.update_layout(xaxis_tickangle=-45)
fig1.show()

# 2. TOP 10 SẢN PHẨM BÁN CHẠY
top10_products = df_transformed.groupBy("Description") \
    .agg(sum("TotalAmount").alias("TotalRevenue")) \
    .orderBy(desc("TotalRevenue")) \
    .limit(10)

pdf_prod = top10_products.toPandas()

fig2 = px.bar(pdf_prod, x="Description", y="TotalRevenue",
              title="🏆 Top 10 Sản phẩm bán chạy nhất",
              text="TotalRevenue",
              labels={"TotalRevenue": "Doanh thu (£)", "Description": "Sản phẩm"})
fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig2.update_layout(xaxis_tickangle=-45, height=600)
fig2.show()
