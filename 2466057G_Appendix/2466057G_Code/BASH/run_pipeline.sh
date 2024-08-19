#!/bin/bash -l
#SBATCH --account=none
#SBATCH --job-name=py-pipeline  # create a short name for your job
#SBATCH --partition="nodes"     # we are using CPU nodes - don't change this
#SBATCH --time=0-01:00:00       # how long do we expect this job to run?
#SBATCH --mem=128G              # how much memory do we want?
#SBATCH --nodes=1               # how many nodes do we want?
#SBATCH --ntasks=1              # how many tasks are we submitting?
#SBATCH --cpus-per-task=64      # how many CPUs do we want for each task?
#SBATCH --ntasks-per-node=1     # how many tasks do we want to run on each node?
#SBATCH --mail-type=END         # mail me when my jobs ends
#SBATCH --mail-type=FAIL        # mail me if my jobs fails
#SBATCH --output=/users/2466057g/project/mars_output/%x-%j.out      # name of output file
#SBATCH --error=/users/2466057g/project/mars_output/%x-%j.error     # name of error file
#SBATCH --mail-user=2466057g@student.gla.ac.uk # email address for notifications 

## Defining global variables - with the first being the value from the first argument
MODE="$1"
VALID_MODES=("full" "feature" "subtypes")

####### INPUT VALIDATION #######
## check the input string is a valid mode the python script is expecting
function is_valid_mode {
    local mode="$1"
    
    for valid_mode in "${VALID_MODES[@]}";
    do
        if [[ "${mode}" == "${valid_mode}" ]];
        then
            return 0
        fi
    done
    return 1
}

if is_valid_mode "$MODE";
then
    echo -e "Mode: '${MODE}' appears to be valid! \n"

else
    echo "Mode: '${MODE}' appears to be invalid!"
    exit 1
fi

############# CODE #############

project_home="/users/2466057g/project"
env_home="${project_home}/proj_env"
py_script_home="${project_home}/py"
bash_script_home="${project_home}/bash"

source "${env_home}/bin/activate"

echo -e "Virtual Environment Activated\n"

# ensure the venv is active
if [[ "${VIRTUAL_ENV}" == "" ]]
then

    echo -e "Virtual Environment seems inactive\n"
    exit 1

fi

cd ${py_script_home}

python3 "run_pipeline.py" --mode "${MODE}" \
    && echo -e "Python Script appears to ran without errors\n" \
    || echo -e "Python Script appears to have ran into errors!!\n"

cd ${project_home}

## close the venv
if [[ "${VIRTUAL_ENV}" != "" ]]
then

    deactivate

    echo -e "Virtual Environment Deactivated\n"

fi

bash ${bash_script_home}/concatenate.sh "${MODE}" \
    && echo -e "CSV files were concatenated\n" \
    || echo -e "Error while concatenating\n"

echo "End of Script"
