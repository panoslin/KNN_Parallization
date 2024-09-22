#!/bin/bash

# Define the sizes
sizes=("small" "medium" "large")

# Define the thread counts
threads=(1 2 4 8 16 32 64 128)

# Create the files
for size in "${sizes[@]}"; do
    for thread in "${threads[@]}"; do
        filename="${size}_${thread}_thread.sh"
        echo "#!/bin/bash" > $filename
        echo "# Script for ${size} with ${thread} thread(s)" >> $filename
        chmod +x $filename  # Make the file executable
        echo "Created file: $filename"
    done
done