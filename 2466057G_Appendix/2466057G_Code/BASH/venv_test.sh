#!/bin/bash -l
#SBATCH --account=none
#SBATCH --job-name=py-test      # create a short name for your job
#SBATCH --partition="nodes"     # we are using CPU nodes - don't change this
#SBATCH --time=0-01:00:00       # how long do we expect this job to run?
#SBATCH --mem=4G                # how much memory do we want?
#SBATCH --nodes=1               # how many nodes do we want?
#SBATCH --ntasks=1              # how many tasks are we submitting?
#SBATCH --cpus-per-task=1       # how many CPUs do we want for each task?
#SBATCH --ntasks-per-node=1     # how many tasks do we want to run on each node?
#SBATCH --mail-type=END         # mail me when my jobs ends
#SBATCH --mail-type=FAIL        # mail me if my jobs fails
#SBATCH --output=%x-%j.out      # name of output file
#SBATCH --error=%x-%j.error     # name of error file
#SBATCH --mail-user=2466057g@student.gla.ac.uk # email address for notifications 

#module python3

############# CODE #############

project_home="/users/2466057g/project"
env_home="${project_home}/proj_env"
py_script_home="${project_home}/py"

source "${env_home}/bin/activate"

# ensure the venv is active
if [[ "${VIRTUAL_ENV}" == "" ]]
then

    echo "Virtual Environment seems inactive"
    exit 1

fi


python3 "${py_script_home}/mars_test.py"


## close the venv
if [[ "${VIRTUAL_ENV}" != "" ]]
then

    deactivate
fi
