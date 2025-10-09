#!/bin/bash
set -eo pipefail  # Exit immediately on error and propagate error codes

# Function: Check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: Command $1 not found. Please install it first."
        exit 1
    fi
}

# Function: Run Python script and check results
run_script() {
    local script_name=$1
    local script_args=$2
    
    echo "Running ${script_name}..."
    echo "Command: python ${script_name} ${script_args}"
    
    # Execute script
    if python "${script_name}" ${script_args}; then
        echo "${script_name} executed successfully"
    else
        echo "Error: ${script_name} failed with exit code $?"
        exit 1
    fi
}

# Main script
generate() {
    local DATASET=$1
    local MODE=$2

    # Check required commands
    echo "Checking required commands..."
    check_command "python"

    # Validate dataset parameter
    echo "Validating parameters..."
    if [[ "$DATASET" != "spider" && "$DATASET" != "bird" && "$DATASET" != "ehr" ]]; then
        echo "Error: Dataset must be either 'spider' or 'bird', current value: $DATASET"
        exit 1
    fi

    # Validate mode parameter
    if [[ "$MODE" != "train" && "$MODE" != "dev" ]]; then
        echo "Error: Mode must be either 'train' or 'dev', current value: $MODE"
        exit 1
    fi

    # Display current configuration
    echo "Starting dataset processing"
    echo "Dataset: $DATASET"
    echo "Mode: $MODE"

    # Execute processing steps
    run_script "preprocess/generate.py" "--dataset $DATASET --mode $MODE"
    run_script "preprocess/check_status.py" "--dataset $DATASET --mode $MODE"
    run_script "preprocess/remove_syntax_error.py" "--dataset $DATASET --mode $MODE"
    run_script "preprocess/sql_to_plan.py" "--dataset $DATASET --mode $MODE"

    echo "All processing steps completed successfully!"

}


for DATASET in bird spider ehr
do
    for MODE in dev train
    do
        generate "$DATASET" "$MODE"
    done
done

exit 0
