import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import subprocess
import random

def count_rows_unix(filename):
    # Function to count the rows on a file
    result = subprocess.run(['wc', '-l', filename], text=True, capture_output=True)
    return int(result.stdout.split()[0])

def load_and_preprocess(csv_files, chunksize=100000):
    # Lists to hold benign and attack data
    benign_data = []
    attack_data = []
    
    # Counter for chunks
    count = 0

    # Loop over all the passed CSV files
    for file in csv_files:
        # Control the sampling rate to give roughly 70K samples from each file based on its size
        sampling_fraction = 70000.0 / count_rows_unix(file)
        # Loop over the CSV and grab chunks
        for chunk in pd.read_csv(file, chunksize=chunksize, low_memory=False):
            # Increment chunk counter, display
            count += 1
            print("On chunk number: " + str(count))

            chunk.drop('Unnamed: 0', axis=1, inplace=True)
            chunk.drop('Flow ID', axis=1, inplace=True)
            chunk.drop(' Source IP', axis=1, inplace=True)
            chunk.drop(' Source Port', axis=1, inplace=True)
            chunk.drop(' Destination IP', axis=1, inplace=True)
            chunk.drop(' Destination Port', axis=1, inplace=True)
            chunk.drop('  Timestamp', axis=1, inplace=True)

            chunk[' Label'] = chunk[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

            # Separate BENIGN and attack data
            benign_chunk = chunk[chunk[' Label'] == 0]
            attack_chunk = chunk[chunk[' Label'] != 1]
            
            # Sample non-BENIGN data if necessary
            if len(attack_chunk) > 1:
                attack_chunk = attack_chunk.sample(frac=sampling_fraction, random_state=random.randint(0, 10000))
            
            # Append the chunks
            benign_data.append(benign_chunk)
            attack_data.append(attack_chunk)
    
    # Concatenate all chunks into two DataFrames
    benign_data = pd.concat(benign_data, ignore_index=True)
    attack_data = pd.concat(attack_data, ignore_index=True)
    
    return benign_data, attack_data


def split_and_save(data, train_file, test_file, test_size=0.2):
    # Split data into training and testing sets
    train, test = train_test_split(data, test_size=test_size, random_state=random.randint(0, 10000))
    
    # Save to CSV
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

def process_files(csv_files):
    # Extract benign and attack data
    benign_data, attack_data = load_and_preprocess(csv_files)
    
    # Save processed files to binary classification folder
    split_and_save(benign_data, '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/train_benign.csv', '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/test_benign.csv')
    split_and_save(attack_data, '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/train_attack.csv', '/s/bach/b/class/cs535/cs535b/binaryclassificationdataset/test_attack.csv')

# List of CSV file paths
csv_files = [
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_DNS.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_LDAP.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_MSSQL.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_NTP.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_NetBIOS.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_SNMP.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_SSDP.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/DrDoS_UDP.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/Syn.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/TFTP.csv',
    '/s/bach/b/class/cs535/cs535b/01-12/UDPLag.csv',
    '/s/bach/b/class/cs535/cs535b/03-11/LDAP.csv',
    '/s/bach/b/class/cs535/cs535b/03-11/MSSQL.csv',
    '/s/bach/b/class/cs535/cs535b/03-11/NetBIOS.csv',
    '/s/bach/b/class/cs535/cs535b/03-11/Portmap.csv',
    '/s/bach/b/class/cs535/cs535b/03-11/Syn.csv',
    '/s/bach/b/class/cs535/cs535b/03-11/UDP.csv',
    '/s/bach/b/class/cs535/cs535b/03-11/UDPLag.csv'
]

# Run the processing
process_files(csv_files)
