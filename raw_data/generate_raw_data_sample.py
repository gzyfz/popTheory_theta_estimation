import subprocess
import re

def parse_msms_output(output, num_sequences):
    # Split the output by lines
    lines = output.split('\n')
    # Find the line indices where the sequences start
    seq_start_indices = [i for i, line in enumerate(lines) if re.match(r'//', line)]
    sequences = []  # List to hold all sequences
    for start in seq_start_indices:
        # Look for the actual sequence lines after 'segsites: <number>'
        segsites_line = lines[start+1]  # Typically, the line after '//' is 'segsites: <number>'
        segsites = int(segsites_line.strip().split()[1])  # Extract the number of segregating sites
        
        if segsites == 0:
            # If there are no segregating sites, all sequences will be identical and can be skipped or handled accordingly
            continue
        # The actual sequences start after the positions of the segregating sites
        pos_line_index = start+2  # Typically, the line after 'segsites: <number>' shows the positions
        seq_start_index = pos_line_index + 1  # Sequences start after the positions line
        
        # Extract the sequences
        for seq_line in lines[seq_start_index:seq_start_index+num_sequences]:
            sequence = seq_line.strip()  # The sequence is the entire line
            if sequence:
                sequences.append(sequence)
            else:
                print(f"Unexpected format or empty line at index {seq_line_index}: '{seq_line}'")
    return sequences



# Simulate with msms and capture the output
theta = 2.0  # Example theta value
num_sequences = 100  # Number of sequences to simulate
command = f"../msms/bin/msms -ms {num_sequences} 1 -t {theta} -r 50 1000"
'''
so every sequence is the what the heplotype should be under the theta={theta}?? is that right??
'''

result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)

# Decode the output from bytes to string
msms_output = result.stdout.decode('utf-8')
print(result)
# Parse the output to get sequences
haplotype_sequences = parse_msms_output(msms_output, num_sequences)

# Output the sequences to a file with the corresponding theta value
with open('msms_training_data.csv', 'w') as file:
    # Write header
    file.write('theta,sequence\n')
    # Write each sequence and theta value
    for sequence in haplotype_sequences:
        file.write(f"{theta},{sequence}\n")

