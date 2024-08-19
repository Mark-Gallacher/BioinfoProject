#!/usr/bin/bash

deactivate() {

    ## ensure we are in the project directory
    cd "/users/2466057g/project/"

    ## if VIRTUAL_ENV is empty, we don't need to deactivate anything
    if [[ "$VIRTUAL_ENV" != "" ]]
    then
        
        deactivate

    fi

}

## run the function
deactivate
