import sys


input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file,'r') as f:
    lines = f.read()
    output_lines = [(', '.join(line.split("\t")[1:7]))+"\n" for line in lines.splitlines() if '#' not in line]
    with open(output_file,'w') as f_out:
        f_out.writelines(output_lines)


