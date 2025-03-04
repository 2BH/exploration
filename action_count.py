import csv

column_sums = {}
algo = "DEIR_end"
action_count_file = f'action_count_{algo}.csv'
n_steps = 6

with open(action_count_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    column_sums = [{header: 0 for header in headers} for _ in range(n_steps)]

    for i, row in enumerate(reader):
        ind_step = int(i/10)
        for j, value in enumerate(row):
            if value:
                column_sums[ind_step][headers[j]] += float(value)

with open(f'action_sum_{algo}.csv', 'a') as csvsum:
    writer = csv.writer(csvsum)
    for i in range(n_steps):
        writer.writerow([str(column_sums[i][headers[0]]/10), str(column_sums[i][headers[1]]/10)])

# for header, total in column_sums.items():
#     print(f"Sum of column '{header}': {total}")

# with open('action_count.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)