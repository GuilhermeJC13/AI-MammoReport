import os
import pandas as pd
"""
Tenho que pegar as bases de reports e imagens localizados na pasta mimic-cxr, rodar o labelizer no report, especificamente nas abas de Impression e Findings
em seguida, devo tratar os textos, separar em partes especificas, remover o que necessário e salvar os textos com mais de 3 palavras. Em seguida, também salvar
o path da imagem correspondente. Por fim, é necessário trocar os valores NaN retornados pelo modelo com 0
"""
INPUT_IMAGES_PATH = 'mimic-cxr/mimic-cxr-images'
INPUT_REPORTS_PATH = 'mimic-cxr/mimic-cxr-reports'

OUTPUT_REPORT_FILE = "reports.csv"
OUTPUT_FILE = 'mimic-cxr-train-meta.csv'

remove_list_temp = ['p10013569']

def treat_raw_report(report_text, patient, report):
    treated_report_list = []
    related_image_list = []
    
    try:
        images = [image for image in os.listdir(os.path.join(INPUT_IMAGES_PATH, patient, report.split('.')[0])) if 'jpg' in image]
    except:
        print(f"Error on path {os.path.join(INPUT_IMAGES_PATH, patient, report.split('.')[0])}")
        return None, None

    if 'IMPRESSION:' in report_text:
        impression = report_text.split('IMPRESSION:')[1]
        for impression_sentence in impression.split('.'):
            if len(impression_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(':  ', '').split(' ')) >= 3 and not '___' in impression_sentence and not 'NOTIFICATION:' in impression_sentence and not 'RECOMMENDATION(S):' in impression_sentence:
                clean_report = impression_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(':  ', '')
                for image in images:
                    treated_report_list.append(clean_report)
                    related_image_list.append(os.path.join(INPUT_IMAGES_PATH, patient, report.split('.')[0], image))

    if 'FINDINGS:' in report_text:
        findings = report_text.split('FINDINGS:')[1]
        findings = findings.split('IMPRESSION:')[0]
        for finding_sentence in findings.split('.'):
            if len(finding_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(':  ', '').split(' ')) >= 3 and not '___' in finding_sentence and not 'NOTIFICATION:' in finding_sentence and not 'RECOMMENDATION(S):' in finding_sentence:
                clean_report = finding_sentence.strip().replace('\n', ' ').replace(',', '').replace('  ', ' ').replace(': ', '')
                for image in images:
                    treated_report_list.append(clean_report)
                    related_image_list.append(os.path.join(INPUT_IMAGES_PATH, patient, report.split('.')[0], image))

    return treated_report_list, related_image_list

def labeler(treated_report_list):
    df = pd.DataFrame(treated_report_list)
    df.to_csv(OUTPUT_REPORT_FILE, index=False, header=False, sep=',')

    os.system("sudo docker build -t chexpert-labeler:latest .")
    os.system("sudo docker run -v $(pwd):/data chexpert-labeler:latest   python label.py --reports_path /data/reports.csv --output_path /data/labeled_reports_test.csv --verbose")
    os.system("sudo chmod 777 labeled_reports_test.csv")

    df = pd.read_csv("labeled_reports_test.csv")
    df = df.rename(columns={"Reports":"report"})

    return df

def treat_labeler(df, related_image_list):
    df.fillna(0, inplace=True)
    df.insert(0, 'imgpath', related_image_list)
    df.to_csv(OUTPUT_FILE, index=True, header=True, sep=',')

def main():
    treated_report_list = []
    related_image_list = []
    for patient_batch in os.listdir(INPUT_REPORTS_PATH):
        for patient in os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch)):
            if patient in remove_list_temp:
                continue
            for report in os.listdir(os.path.join(INPUT_REPORTS_PATH, patient_batch, patient)):
                with open(os.path.join(INPUT_REPORTS_PATH, patient_batch, patient, report)) as report_file:
                    report_text = report_file.read()
                    treated_report, related_image = treat_raw_report(report_text, patient, report)
                    if treated_report == None:
                        continue
                    related_image_list = [*related_image_list, *related_image]
                    treated_report_list = [*treated_report_list, *treated_report]
                    
    df = labeler(treated_report_list)
    treat_labeler(df, related_image_list)

if __name__ == '__main__':
    main()