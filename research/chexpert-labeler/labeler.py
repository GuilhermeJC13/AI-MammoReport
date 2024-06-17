import os
import pandas as pd
import random
import math
"""
Tenho que pegar as bases de reports e imagens localizados na pasta mimic-cxr, rodar o labelizer no report, especificamente nas abas de Impression e Findings
em seguida, devo tratar os textos, separar em partes especificas, remover o que necessário e salvar os textos com mais de 3 palavras. Em seguida, também salvar
o path da imagem correspondente. Por fim, é necessário trocar os valores NaN retornados pelo modelo com 0
"""
INPUT_IMAGES_PATH = 'mimic-cxr/mimic-cxr-images'
INPUT_REPORTS_PATH = 'mimic-cxr/mimic-cxr-reports'

OUTPUT_REPORT_FILE = "reports.csv"
OUTPUT_TRAIN_FILE = 'mimic-cxr-train-meta.csv'
OUTPUT_VALIDATION_FILE1 = 'chexpert-5x200-val-meta.csv'
OUTPUT_VALIDATION_FILE2 = 'sentence-label.csv'

valid_patient_batch = ['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19']
remove_list_temp = ['p10013569']
num_groups = 10
num_labels = 14

def treat_raw_report(report_text, patient_batch, patient, report):
    treated_report_list = []
    related_image_list = []
    
    try:
        images = [image for image in os.listdir(os.path.join(INPUT_IMAGES_PATH, patient_batch, patient, report.split('.')[0])) if 'jpg' in image]
    except:
        print(f"Error on path {os.path.join(INPUT_IMAGES_PATH, patient_batch, patient, report.split('.')[0])}")
        return None, None

    if 'IMPRESSION:' in report_text:
        impression = report_text.split('IMPRESSION:')[1]
        for impression_sentence in impression.split('.'):
            if len(impression_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(':  ', '').split(' ')) >= 3 and not '___' in impression_sentence and not 'NOTIFICATION:' in impression_sentence and not 'RECOMMENDATION(S):' in impression_sentence:
                clean_report = impression_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(':  ', '')
                for image in images:
                    treated_report_list.append(clean_report)
                    related_image_list.append(os.path.join(INPUT_IMAGES_PATH, patient_batch, patient, report.split('.')[0], image))

    if 'FINDINGS:' in report_text:
        findings = report_text.split('FINDINGS:')[1]
        findings = findings.split('IMPRESSION:')[0]
        for finding_sentence in findings.split('.'):
            if len(finding_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(':  ', '').split(' ')) >= 3 and not '___' in finding_sentence and not 'NOTIFICATION:' in finding_sentence and not 'RECOMMENDATION(S):' in finding_sentence:
                clean_report = finding_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(': ', '')
                for image in images:
                    treated_report_list.append(clean_report)
                    related_image_list.append(os.path.join(INPUT_IMAGES_PATH, patient_batch, patient, report.split('.')[0], image))

    return treated_report_list, related_image_list

def labeler(treated_report_list):
    df = pd.DataFrame(treated_report_list)
    df.to_csv(OUTPUT_REPORT_FILE, index=False, header=False, sep=',')

    os.system('sudo docker build -t chexpert-labeler:latest .')
    os.system('sudo docker run -v "$(pwd):/data" chexpert-labeler:latest   python label.py --reports_path /data/reports.csv --output_path /data/labeled_reports_test.csv --verbose')
    os.system('sudo chmod 777 labeled_reports_test.csv')

    df = pd.read_csv("labeled_reports_test.csv")
    df = df.rename(columns={"Reports":"report"})

    return df

def treat_labeler(df, related_image_list):
    df.fillna(0, inplace=True)
    df.insert(0, 'imgpath', related_image_list)

def create_train_and_validation_files(df):

    num_objects_labels = [(df['Enlarged Cardiomediastinum'] == 1).sum(), (df['Cardiomegaly'] == 1).sum(), (df['Lung Lesion'] == 1).sum(),
                          (df['Lung Opacity'] == 1).sum(), (df['Edema'] == 1).sum(), (df['Consolidation'] == 1).sum(),
                          (df['Pneumonia'] == 1).sum(), (df['Atelectasis'] == 1).sum(), (df['Pneumothorax'] == 1).sum(),
                          (df['Pleural Effusion'] == 1).sum(), (df['Pleural Other'] == 1).sum(), (df['Fracture'] == 1).sum(),
                          (df['Support Devices'] == 1).sum(), (df['No Finding'] == 1).sum()]
    num_sample_label = math.floor(min(num_objects_labels)/2)
    count_labels_added = [0] * num_labels                       

    df = df.sample(frac=1).reset_index(drop=True)              
    df_validation = pd.DataFrame(columns=df.columns)                             
    for index, row in df.iterrows():
        if row['Enlarged Cardiomediastinum'] == 1 and count_labels_added[0] < num_sample_label:
            count_labels_added[0] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Cardiomegaly'] == 1 and count_labels_added[1] < num_sample_label:
            count_labels_added[1] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Lung Lesion'] == 1 and count_labels_added[2] < num_sample_label:
            count_labels_added[2] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Lung Opacity'] == 1 and count_labels_added[3] < num_sample_label:
            count_labels_added[3] += 1
            df_validation.loc[len(df)] = row
            df = df.drop(index)
        elif row['Edema'] == 1 and count_labels_added[4] < num_sample_label:
            count_labels_added[4] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Consolidation'] == 1 and count_labels_added[5] < num_sample_label:
            count_labels_added[5] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Pneumonia'] == 1 and count_labels_added[6] < num_sample_label:
            count_labels_added[6] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Atelectasis'] == 1 and count_labels_added[7] < num_sample_label:
            count_labels_added[7] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Pneumothorax'] == 1 and count_labels_added[8] < num_sample_label:
            count_labels_added[8] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Pleural Effusion'] == 1 and count_labels_added[9] < num_sample_label:
            count_labels_added[9] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Pleural Other'] == 1 and count_labels_added[10] < num_sample_label:
            count_labels_added[10] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Fracture'] == 1 and count_labels_added[11] < num_sample_label:
            count_labels_added[11] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['Support Devices'] == 1 and count_labels_added[12] < num_sample_label:
            count_labels_added[12] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif row['No Finding'] == 1 and count_labels_added[13] < num_sample_label:
            count_labels_added[13] += 1
            df_validation.loc[len(df_validation)] = row
            df = df.drop(index)
        elif min(count_labels_added) == num_sample_label:
            break

    df_chexpert_5x200_val_meta = df_validation.drop(columns=['report'])
    df_setence_label = df_validation.drop(columns=['imgpath'])

    df_chexpert_5x200_val_meta.to_csv(OUTPUT_VALIDATION_FILE1, index=True, header=True, sep=',')
    df_setence_label.to_csv(OUTPUT_VALIDATION_FILE2, index=True, header=True, sep=',')
    df.to_csv(OUTPUT_TRAIN_FILE, index=True, header=True, sep=',')       

def main():

    global_df = pd.DataFrame()
    global_treated_report_list = []
    global_related_image_list = []

    treated_report_list = []
    related_image_list = []

    for patient_batch in os.listdir(INPUT_REPORTS_PATH):
        if patient_batch in valid_patient_batch:

            tam_groups = math.ceil(len(os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch))) / num_groups)
            last_patient_group = tam_groups - 1

            index_patient = 0
            while index_patient != len(os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch))):

                patient = os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch))[index_patient]

                if patient in remove_list_temp:
                    if index_patient == last_patient_group or index_patient == len(os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch)))-1:
                        last_patient_group += tam_groups

                        global_related_image_list += related_image_list
                        global_treated_report_list += treated_report_list

                        df = labeler(treated_report_list)
                        related_image_list.clear()
                        treated_report_list.clear()

                        global_df = pd.concat([global_df, df])

                    index_patient += 1  
                    continue

                for report in os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch, patient)):
                    with open(os.path.join(INPUT_REPORTS_PATH, patient_batch, patient, report)) as report_file:
                        report_text = report_file.read()
                        treated_report, related_image = treat_raw_report(report_text, patient_batch, patient, report)
                        if treated_report == None:
                            continue
                        related_image_list = [*related_image_list, *related_image]
                        treated_report_list = [*treated_report_list, *treated_report] 

                if index_patient == last_patient_group or index_patient == len(os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch)))-1:
                    last_patient_group += tam_groups

                    global_related_image_list += related_image_list
                    global_treated_report_list += treated_report_list

                    df = labeler(treated_report_list)
                    related_image_list.clear()
                    treated_report_list.clear()

                    global_df = pd.concat([global_df, df])

                index_patient += 1  
    
    treat_labeler(global_df, global_related_image_list)
    create_train_and_validation_files(global_df)

if __name__ == '__main__':
    main()