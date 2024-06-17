import pandas as pd

def define_class(row):
    for i, col in enumerate(df.columns):
        if row[col] == 1:
            return i

df = pd.read_csv('mimic-cxr-train-meta.csv')

aux1 = df['imgpath']
aux2 = df['report']
df = df.drop(columns=['imgpath','report'])
df['class'] = df.apply(define_class, axis=1)
df['imgpath'] = aux1
df['report'] = aux2

valores_para_manter = [3, 6, 7, 9, 11]
df_filtrado = df[df['class'].isin(valores_para_manter)]
df_filtrado.drop(columns=['class'])

colunas_para_manter = ['imgpath', 'report', 'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
df_reduzido = df_filtrado.loc[:, colunas_para_manter]

print(df_reduzido.value_counts())
df_reduzido.sample(frac=1, random_state=42)
df_reduzido.to_csv("mimic_test.csv")
