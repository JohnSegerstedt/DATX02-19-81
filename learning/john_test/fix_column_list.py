
result = ""

for row in open("all_columns.txt", "r"):
    split = row.split(',')
    result = "'" + str(split) + "'"

print(result)
