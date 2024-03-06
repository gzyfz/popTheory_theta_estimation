import subprocess
import numpy as np
import os
# Define parameters for msms
sample_size = 100
replicates = 1
theta_values = [i/5 for i in range(1,100,1)]  # Define your theta values here
N = 1e6  # Effective population size

# Function to run msms and get the output
def run_msms(theta, sample_size, replicates, N):
    command = f"../msms/bin/msms -ms {sample_size} {replicates} -t {theta} -N {N}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if stderr:
        raise Exception(f"Error in msms run: {stderr.decode('utf-8')}")
        
    return stdout.decode('utf-8')

# Function to parse msms output and extract phased SNP matrix
def parse_msms_output(msms_output):
    lines = msms_output.strip().split('\n')
    snp_matrix = []
    parsing_snps = False  # Flag to indicate if we are currently parsing SNP lines

    for line in lines:
        if line.startswith('//'):  # Start of a new simulation replicate
            parsing_snps = False  # Reset the flag as we encountered a new replicate
            snp_matrix = []  # Reset for next replicate
        elif 'segsites:' in line:
            parsing_snps = True  # Next lines will contain SNPs
            if 'segsites: 0' in line:  # No segregating sites, hence no SNP matrix to be made
                return None
        elif parsing_snps and all(char in '01' for char in line.strip()):  # Ensure the line contains only '0' or '1'
            snp_line = list(map(int, line.strip()))  # Convert each character digit to an integer
            snp_matrix.append(snp_line)
    
    return np.array(snp_matrix, dtype=int) if snp_matrix else None  # Return the last replicate

# Main loop to run simulations and create SNP matrices for different thetas
all_snp_matrices = []
max_width = 0  # Variable to keep track of the maximum width

for index, theta in enumerate(theta_values):
    if index % 10 == 0:
        print(index, "samples have been created")
    msms_output = run_msms(theta, sample_size, replicates, N)
    snp_matrix = parse_msms_output(msms_output)
    
    if snp_matrix is not None:
        all_snp_matrices.append(snp_matrix)
        max_width = max(max_width, snp_matrix.shape[1])

# Pad each matrix to have the maximum width
padded_snp_matrices = [np.pad(matrix, ((0, 0), (0, max_width - matrix.shape[1])), 'constant', constant_values=0)
                       if matrix is not None else None
                       for matrix in all_snp_matrices]

# Convert padded matrices to numpy array
padded_snp_matrices = [matrix for matrix in padded_snp_matrices if matrix is not None]  # Filter out None values
all_snp_matrices_array = np.array(padded_snp_matrices)



output_dir = 'snp_matrices'
os.makedirs(output_dir, exist_ok=True)

# Save the SNP matrices and theta values
for i, matrix in enumerate(all_snp_matrices_array):
    output_path = os.path.join(output_dir, f'snp_matrix_theta{theta_values[i]}.npy')
    np.save(output_path, matrix)

