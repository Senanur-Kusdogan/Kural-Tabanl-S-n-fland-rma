import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("dataset/persona.csv")
print(df)
df.head()
df.info()
# Kaç unique SOURCE vardır, frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Hangi PRICE'dan kaçar tane satış gerçekleşmiştir?
df["PRICE"].value_counts()

# Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Source türlerine göre satış sayıları nedir?
df.groupby("SOURCE")["PRICE"].count()
df["SOURCE"].value_counts()

# Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": "mean"})
df.groupby("COUNTRY")["PRICE"].mean()

# SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()

# COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
# df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"], "survived": "mean", "sex": "count"})
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

# Çıktıyı PRICE'a göre sıralayın ve agg_df olarak kaydedin
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

# Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df = agg_df.reset_index()
agg_df.head()

# age değişkenini kateorik değişkene çevir ve agg_ df'e ekle
# aralık oluştur
# df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
agg_df["AGE"] = agg_df["AGE"].astype("category")
agg_df["AGE_cat"] = pd.cut(agg_df["AGE"], [0, 18, 23, 31, 41, 70], labels=["0_18", "19_23", "24_30", "31_40", "41_70"])
agg_df.head()

# Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
agg_df["customers_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_cat"]].agg(lambda x: "_".join(x).upper(), axis=1)
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()
agg_df.head()

# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre groupby yapıp price mean, max, sum’larını alınız).
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head()

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmenteaittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmenteaittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
