import numpy as np
import csv

# Define class
class Statistics():
    def __init__(self, conn_head_sem_class):
        self.conn_head_sem_class = conn_head_sem_class
        self.frequency = 0

# Init param
result = []  # Statistic result
keylist = []  # Primary key for index
conn_words = []  # Set connectives
conn_words_count = []  # Set connectives count

# Get statistics
def run_statistics(conn, conn_head_sem_class):
    if conn_head_sem_class not in keylist:
        keylist.append(conn_head_sem_class)
        new_situation = Statistics(conn_head_sem_class)
        result.append(new_situation)
    result[keylist.index(conn_head_sem_class)].frequency += 1

# Load dataset
with open ('data/pdtb2.csv', 'r', newline='') as f:
    csv_reader = csv.reader(f,delimiter=',')
    # For each instance (row)
    for row in csv_reader:
        if row[0] == 'Relation':
            continue
        # ConnHead
        if row[9] != '':
            conn = row[9]
            for index in range(12,14):
                if row[index] != '':
                    conn_head_sem_class = transform_sem_class(row[index])
                    run_statistics(conn, conn_head_sem_class)
        # Conn1
        if row[10] != '':
            conn = row[10]
            for index in range(12,14):
                if row[index] != '':
                    conn_head_sem_class = transform_sem_class(row[index])
                    run_statistics(conn, conn_head_sem_class)
        # Conn2
        if row[11] != '':
            conn = row[10]
            for index in range(14,16):
                if row[index] != '':
                    conn_head_sem_class = transform_sem_class(row[index])
                    run_statistics(conn, conn_head_sem_class)

# Write result in csv
with open ('result/statistics.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)

    csv_writer.writerow(['conn_head_sem_class', 'frequency'])
    for item in result:
        csv_writer.writerow([item.conn_head_sem_class, item.frequency])