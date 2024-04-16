import csv
import pandas as pd

# Mo file goc ( chua qua xu ly )
# Doc file va ghi vao newlines
with open('Dataset.data') as input_f:
    lines = input_f.readlines()
    newlines = []
    for line in lines:
        newLine = line.strip().split()
        newlines.append(newLine)

# Ghi vao file data.csv (tao san)
# Them header (column names)
with open('data.csv', 'w', newline='') as output_f:
    cont_h = ['Pclass', 'Mature', 'Gender', 'Survived']
    file_writer = csv.writer(output_f)
    file_writer.writerow(cont_h)
    file_writer.writerows(newlines)

# Thay cac gia tri chu cai thanh gia tri so
def mapping_value(data):
    gender = {"male": 1, "female": 0}
    data['Gender'] = data['Gender'].map(gender)
    pclass = {"1st":1,"2nd":2,"3rd":3,'crew':4}
    data['Pclass'] = data['Pclass'] .map(pclass)
    survied = {"yes":1,"no":0}
    data['Survived'] = data['Survived'].map(survied)
    mature = {"adult":1,"child":0}
    data['Mature'] = data['Mature'].map(mature)

# Xuat file ra console kiem tra noi dung
df = pd.read_csv('data.csv', index_col=None)
print('Dataset : ')
print(df.head())
print(df.tail())


