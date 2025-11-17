import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("/Users/kayvans/Documents/mimic-demo/cd-pass2/draft.csv")


df = df.drop_duplicates(subset=['hadm_id'])
df = df.drop(columns=["hadm_id","subject_id","stay_id","admittime","dischtime","end_window","start_window","intime","outtime","first_antibiotic_time","uti","temperature_max_C"], axis=1)
categorical_cols= ['gender', 'race', 'admission_type', 'admission_location']


binary_cols = ['vent_or_intubation', 'sepsis', 'septic_shock', 'hospital_expire_flag','arf','antibiotics_given','vaso_given']
cont_cols = df.select_dtypes(include=['float', 'int']).columns.difference(binary_cols)
df[cont_cols] = SimpleImputer(strategy='median').fit_transform(df[cont_cols])

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes
for col in binary_cols:
    df[col] = df[col].astype('int')

core_cols = ['antibiotics_given', 'vaso_given','vent_or_intubation', 'creatinine_admission_max', 'bun_admission_max', 'blood_pressure_min','lactate_max','anchor_age','gender','hospital_expire_flag', 'septic_shock', 'sepsis', 'arf']
data = df[core_cols].to_numpy()
cg = pc(data, indep_test='fisherz')
nodes = cg.G.get_nodes()
bk = BackgroundKnowledge()
for i in range(len(nodes)):
    bk.add_forbidden_by_node(nodes[9], nodes[i])
bk.add_forbidden_by_node(nodes[10], nodes[11])

cg_with_background_knowledge = pc(data, background_knowledge=bk)
cg_with_background_knowledge.draw_pydot_graph(labels=core_cols)
pyd = GraphUtils.to_pydot(cg_with_background_knowledge.G,labels=core_cols) 
pyd.write_png("model1_fisherz_imputing.png")
df[core_cols].to_csv("data_processed.csv", index=False)