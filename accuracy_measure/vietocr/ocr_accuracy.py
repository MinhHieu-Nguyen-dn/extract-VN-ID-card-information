import pandas as pd
import nlptutti as metrics

path_to_reference = 'accuracy_measure/vietocr/reference_text.csv'
data = pd.read_csv(path_to_reference)

# Get lower text case in refs and preds text
full_text_lower = []
ref_text_lower = []

for i in range(len(data)):
    full_text_lower.append(data['full_text'][i].lower())
    ref_text_lower.append(data['Reference_text'][i].lower())

# Calculate CER and WER then insert to the dataframe
mean_cer = 0
cer = []
wer = []
for i in range(len(full_text_lower)):
    cer.append(round((metrics.get_cer(ref_text_lower[i],full_text_lower[i])['cer'] * 100),2))
    wer.append(round((metrics.get_wer(ref_text_lower[i], full_text_lower[i])['wer'] * 100),2))
    
data.insert(3, 'cer', cer, True)
data.insert(4, 'wer', wer, True)

# Calculate mean of CER and WER and accuracy of the model
mean_cer = 0
mean_wer = 0
for i in range(len(data)):
    mean_cer = mean_cer + data['cer'][i]
    mean_wer = mean_wer + data['wer'][i]
mean_cer = mean_cer / len(data)
mean_wer = mean_wer / len(data)

print(f'Mean CER = {round(mean_cer,2)}%, Mean WER = {round(mean_wer,2)}%')
accuracy = 100 - round(mean_cer)
if accuracy >= 98:
    print(f'----Accuracy = {accuracy}%-----\n----Good OCR accuracy----')
elif accuracy < 90:
    print(f'----Accuracy = {accuracy}%-----\n----Poor OCR accuracy----')
else:
    print(f'----Accuracy = {accuracy}%-----\n----Average OCR accuracy----')

path_output = 'accuracy_measure/vietocr/ocr_accuracy.csv'
data.to_csv(path_output)
