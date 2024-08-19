#!/usr/bin/bash

activate() {

    if [[ "$VIRTUAL_ENV" == "" ]]
    then 
        ## ensure we are in the project directory
        cd "/users/2466057g/project/"

        ## activate the python env.
        source "/users/2466057g/project/proj_env/bin/activate"

    fi
}

## run the function
activate
