import csv
"""
convert libsvm file to csv'
"""

input_file = '../a9a/a9a.test'
output_file = '../a9a/a9a_test.csv'
d = 123

reader = csv.reader(open(input_file), delimiter=" ")
writer = csv.writer(open(output_file, 'w'))

for line in reader:
    label = line.pop(0)


    # print line
    line = map(lambda x: tuple(x.split(":")), line[:-1])
    # print line

    new_line = [label] + [0] * d
    for i, v in line:
        i = int(i)
        if i <= d:
            new_line[i] = v

    writer.writerow(new_line)
