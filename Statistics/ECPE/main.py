import csv

# Define class
class Statistics():
    def __init__(self, key, emo_conn, cau_conn, type, dis):
        self.key = key
        self.emo_conn = emo_conn
        self.cau_conn = cau_conn
        self.type = type
        self.dis = dis
        self.frequency = 0

# Init param
result = []  # Statistic result
key = []  # Primary key for index
conn_words = []  # Set connectives
# conn_words_count = []  # Set connectives connt

# Load cause connectives
with open ('../../data/cause_uniconn.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            conn_words.append(word)
        line = f.readline()

with open ('../../data/all_data_pair.txt', 'r', encoding='utf-8') as f:  # Encode by utf-8 for Chinese
    sec = f.readline()  # Read section ID and length

    # For each section
    while sec:

        # Get section length
        num = sec.split(' ')
        length = int(num[1])
        content = ['' for i in range(length)]
        refined_content = ['' for i in range(length)]

        pairs = f.readline().lstrip().rstrip()  # Get the index of pairs and delete the beginning ' ' and ending '\n'

        # Get the index of pairs (int)
        pairs_index = []
        for pair in pairs.split(', '):
            pairs_index.append(list(map(int, pair.lstrip('(').rstrip(')').split(','))))

        # Get the content of section
        for i in range(length):
            content[i] = f.readline().lstrip().rstrip().split(',')[3]

            # Get the raw content
            for word in content[i].split(' '):
                refined_content[i] += word
        
        # For each pair
        for pair in pairs_index:
            dis = pair[1] - pair[0]  # Calculate dis = cause - emotion

            # Emotion clause
            emo_conn_flag = 0  # Set no conn as default
            emo_conn = ''
            
            for i in range(len(refined_content[pair[1] - 1])):  # Single-character word
                if refined_content[pair[1] - 1][i] in conn_words:
                    emo_conn_flag = 1
                    emo_conn = refined_content[pair[1] - 1][i]
            
            # Cause clause
            cau_conn_flag = 0  # Set no conn as default
            cau_conn = ''

            for i in range(len(refined_content[pair[0] - 1])):  # Single-character word
                if refined_content[pair[0] - 1][i] in conn_words:
                    cau_conn_flag = 1
                    cau_conn = refined_content[pair[0] - 1][i]

            # Judge structure type
            # type0 (emo, cau), type1 (emo, conn, cau), type2 (conn, emo, cau), type3 (conn, emo, conn, cau)
            # We always rewrite the sentence to make sure emo is ahead of cau for our research
            type = 0
            if cau_conn_flag == 1:
                type = 1
            if emo_conn_flag == 1:
                type = 2
            if cau_conn_flag == 1 & emo_conn_flag == 1:
                type = 3

            # Get statistics
            pair_key = emo_conn + cau_conn + str(type) + str(dis)  # Set unique key

            if pair_key not in key:
                key.append(pair_key)
                new_situation = Statistics(pair_key, emo_conn, cau_conn, type, dis)
                result.append(new_situation)
            
            result[key.index(pair_key)].frequency += 1

        sec = f.readline()  # Read following section length

# Write result in csv
with open ('result/result.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)

    csv_writer.writerow(['pair_key', 'emo_conn', 'cau_conn', 'type', 'dis', 'frequency'])
    for item in result:
        csv_writer.writerow([item.key, item.emo_conn, item.cau_conn, item.type, item.dis, item.frequency])