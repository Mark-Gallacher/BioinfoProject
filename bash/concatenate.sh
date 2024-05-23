#!/bin/bash -l

if [ "$#" -eq 0 ]
then
  echo "Please Specify a Mode of analysis by supplying the name of the directory"
  exit 1
fi

## mode of analysis
## full means we used all data
## subtypes means we used only the hypertensive groups
mode=$1

is_dir() {

    local dir=$1

    [[ -d $dir ]]
}

extract_header() {

    local dir=$1
    local out_file=$2
    local tmp_file="${dir}/tmp_extract_header.txt"

    head -n 1 ${dir}/*.csv \
    | grep "^[[:alpha:]]" \
    | uniq > ${tmp_file} 

    [[ -f ${tmp_file} ]] \
        || echo "Unable to generate tmp file"; #exit 1

    file_length=$(
            wc -l ${tmp_file} \
            | cut -d ' ' -f 1
        )

    [[ $file_length -eq 1 ]] \
        && {
            cat $tmp_file > $out_file ; \
            rm ${tmp_file} ; 
        } \
        || {
            echo "csv files have mismatching headers in ${dir}!" ; \
            echo "Headers:" ; \
            echo "${tmp_file}" ; \
            echo "File had length ${file_length}"
            #exit 1 ; 
        }

}


extract_body() {

    local dir=$1
    local out_file=$2
    
    ## get everything apart from the first line
    is_dir "${dir}" \
        || {
            echo "Directory does not seem to exist - ${dir}" ; \
            #exit 1 ; 
            }

    for file in ${dir}/*.csv;
    do
        tail --lines=+2 ${file} \
        | grep "^[[:alpha:]]" >> $out_file
             
    done

}

main() {

    local project_home="/users/2466057g/project"
    local csv_dir="${project_home}/data/${mode}"

    ## check the directory exists before defining the out dir - we don't want it to do in a weird place.
    is_dir $csv_dir \
        && local out_dir="${csv_dir}/${mode}/merged" \
        || echo "Directory does not seem to exist - ${csv_dir}"


    mkdir -p ${out_dir}

    local dir ## init the for loop var.
    for dir in metrics params;
    do
        
        local folder="${csv_dir}/${dir}"
        local out_file="${out_dir}/${dir}_merged.csv"        

        ## check the folder with the csvs actually exists.
        is_dir $folder \
            && {
                extract_header $folder $out_file ; \
                extract_body $folder $out_file ;
            } \
            || {
                echo "Directory does not seem to exist - ${folder}"; \
                #exit 1;
                }


    done
 }

main


