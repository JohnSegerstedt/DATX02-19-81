import csv
import os
import pandas as pandas


# Changable variables
mmr_cutoff = 4400 # ca: top 10k replays
print_out = False
save_csv = True
csv_source_folder_name = "cluster_data"
csv_destination_folder_name = "new_data"

# Do-not-touch variables
number_of_files = 0
number_of_done = 0

# --- MAIN ---
for fileName in os.listdir(csv_source_folder_name):
    number_of_files += 1

for fileName in os.listdir(csv_source_folder_name):
    data = pandas.read_csv(csv_source_folder_name+"\\"+fileName)
    data = data.drop('Unnamed: 0', axis=1)
    newData = data.ix[(data['0P1_mmr'] > mmr_cutoff) | (data['0P2_mmr'] > mmr_cutoff)]
    if print_out:
        print(newData['0P1_mmr'])
    if not os.path.exists(csv_destination_folder_name):
        os.mkdir(csv_destination_folder_name)
    fullname = os.path.join(csv_destination_folder_name, fileName)  
    if save_csv:
        newData.to_csv(fullname, sep=',')
    number_of_done += 1
    percentage = int(100.0 * (number_of_done / number_of_files))
    print("--- "+fileName+" FINISHED | "+str(number_of_done)+"/"+str(number_of_files)+" | "+str(percentage)+"% ---")
print("--- SCRIPT FINISHED ---")
   
        
    

