#!/bin/bash

binaries=("threaded", "openmp", "mpi")
sizes=("small" "medium" "large")
threads=(1 2 4 8 16 32 64 128)

mkdir slurm_scripts
# Create the files
for binary in "${binaries[@]}"; do
    mkdir "slurm_scripts/${binary}"
    for size in "${sizes[@]}"; do
        mkdir "slurm_scripts/${binary}/${size}"
        for thread in "${threads[@]}"; do
            filename="${thread}.sh"
            echo "#!/bin/bash" > slurm_scripts/${binary}/${size}/$filename
            chmod +x $filename  # Make the file executable
            echo "Created file: $filename"
        done
    done
done
