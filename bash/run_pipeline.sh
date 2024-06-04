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

#module python3

############# CODE #############

project_home="/users/2466057g/project"
env_home="${project_home}/proj_env"
py_script_home="${project_home}/py"
bash_script_home="${project_home}/bash"

source "${env_home}/bin/activate"

echo "Virtual Environment Activated"

# ensure the venv is active
if [[ "${VIRTUAL_ENV}" == "" ]]
then

    echo "Virtual Environment seems inactive"
    exit 1

fi

cd ${py_script_home}

python3 "run_pipeline.py" \
    && echo "Python Script appears to ran without errors" \
    || echo "Python Script appears to have ran into errors!!"

cd ${project_home}

## close the venv
if [[ "${VIRTUAL_ENV}" != "" ]]
then

    deactivate

    echo "Virtual Environment Deactivated"

fi

bash ${bash_script_home}/concatenate.sh "feature" \
    && echo "CSV files were concatenated" \
    || echo "Error while concatenating"

echo "End of Script"
