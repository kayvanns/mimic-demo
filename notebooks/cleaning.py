import pandas as pd
from datetime import timedelta

## reading files and creating dataframes for tables

path = "../data/mimic-iv-clinical-database-demo-2.2/hosp/"
path2 = "../data/mimic-iv-clinical-database-demo-2.2/icu/"

files = ["admissions", "patients", "labevents", "d_labitems", "prescriptions","pharmacy","transfers", "diagnoses_icd","d_icd_diagnoses"]
dfs = {}

for name in files:
    dfs[name] = pd.read_csv(path + f"{name}.csv.gz")

admissions = dfs["admissions"]
patients = dfs["patients"]
labs = dfs["labevents"]
d_labitems = dfs["d_labitems"]
prescriptions = dfs["prescriptions"]
pharmacy = dfs["pharmacy"]
transfers=dfs["transfers"]
diagnoses = dfs["diagnoses_icd"]
d_diagnoses = dfs["d_icd_diagnoses"]

files2 = ["icustays", "inputevents", "outputevents", "procedureevents","chartevents", "datetimeevents", "d_items"]
dfs2 = {}

for name in files2:
    dfs2[name] = pd.read_csv(path2 + f"{name}.csv.gz")
icustays = dfs2["icustays"]
inputevents = dfs2["inputevents"]
outputevents = dfs2["outputevents"]
procedureevents = dfs2["procedureevents"]
chartevents = dfs2["chartevents"]
datetimeevents = dfs2["datetimeevents"]
d_items = dfs2["d_items"]

## merges the patients, admissions and icustays table based on the diagnosis
def get_base(title):
    d_title = d_diagnoses[d_diagnoses["long_title"].str.contains(title, case=False,na=False)].copy()
    title = diagnoses[diagnoses["icd_code"].isin(d_title["icd_code"])].copy()
    df1 = patients[patients["subject_id"].isin(title["subject_id"])].copy()
    df2 = admissions[admissions["hadm_id"].isin(title["hadm_id"])].copy()
    df1 = df1.reset_index(drop=True)
    df2=df2.reset_index(drop=True)
    df = df1.merge(df2,on="subject_id")
    col = "hadm_id"
    df = df[[col] + [c for c in df.columns if c != col]]
    df = df.merge(icustays,on=["hadm_id","subject_id"],how="left")
    return df

## converts import dates to datetime objects, and gets ICU + hospital duration
def datetime(df):
    df["outtime"]= pd.to_datetime(df["outtime"])
    df["intime"]=pd.to_datetime(df["intime"])
    df["ICU_length"] = df["outtime"] - df["intime"]
    df["admittime"] =pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["Hospital_length"] = df["dischtime"]-df["admittime"]
    df["end_window"] = (df["intime"] + timedelta(hours=4))
    return df

## collects highest/lowest measurements within 4 hours of ICU admission
def get_chartevents(df):
    df = df.merge(chartevents,on=["hadm_id","subject_id","stay_id"],how="left")
    df["charttime"] =pd.to_datetime(df["charttime"])
    df["storetime"] = pd.to_datetime(df["storetime"])
    mask = (df['charttime'] >= df['intime']) & (df['charttime'] <= df["end_window"])
    return df[mask] 

vitals = {"heart_rate_max":{'itemid':220045, "agg":'max'}, "blood_pressure_min":{'itemid':220181,"agg":'min'}}

def get_vitals(df):
    results = []
    for vital,info in vitals.items():
        temp = (df[df["itemid"]==info["itemid"]].groupby("stay_id")["valuenum"].agg(info["agg"]).reset_index(name=f"{info['agg']}_{vital}"))
        df = df.merge(temp, on="stay_id", how="left")
        break
    return df
    