import numpy as np
import csv
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Transform sem_class into 2-level sem_class
def transform_sem_class(sem_class):
    sem_class_parts = sem_class.split('.')
    if len(sem_class_parts) > 1:
        return sem_class_parts[0] + '.' + sem_class_parts[1]
    else:
        return sem_class_parts[0]

# Init csv
with open ('../../data/dataset.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['relation', 'section', 'filenumber', 'Arg1_RawText', 'Arg2_RawText', 'conn', 'conn_head_sem_class'])

# Write result in csv
def write_pair(relation, section, filenumber, Arg1_RawText, Arg2_RawText, conn, conn_head_sem_class):
    with open ('../../data/dataset.csv', 'a', newline='') as f:
        if len(tokenizer.tokenize(conn)) == 1:
            csv_writer = csv.writer(f)
            csv_writer.writerow([relation, section, filenumber, Arg1_RawText, Arg2_RawText, conn, conn_head_sem_class])

# Load dataset
with open ('../../data/pdtb2.csv', 'r', newline='') as f:
    csv_reader = csv.reader(f,delimiter=',')
    # For each instance (row)
    for row in csv_reader:
        if row[0] == 'Relation':
            continue
        # relation-0, section-1, filenumber-2, Arg1_RawText-27, Arg2_RawText-39
        relation = row[0]
        section = row[1]
        filenumber = row[2]
        Arg1_RawText = row[27]
        Arg2_RawText = row[39]
        # ConnHead
        if row[9] != '':
            conn = row[9]
            for index in range(12,14):
                if row[index] != '':
                    conn_head_sem_class = transform_sem_class(row[index])
                    write_pair(relation, section, filenumber, Arg1_RawText, Arg2_RawText, conn, conn_head_sem_class)
        # Conn1
        if row[10] != '':
            conn = row[10]
            for index in range(12,14):
                if row[index] != '':
                    conn_head_sem_class = transform_sem_class(row[index])
                    write_pair(relation, section, filenumber, Arg1_RawText, Arg2_RawText, conn, conn_head_sem_class)
        # Conn2
        if row[11] != '':
            conn = row[10]
            for index in range(14,16):
                if row[index] != '':
                    conn_head_sem_class = transform_sem_class(row[index])
                    write_pair(relation, section, filenumber, Arg1_RawText, Arg2_RawText, conn, conn_head_sem_class)
