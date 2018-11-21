#!/usr/bin/bash
# split json by the years
function num_lines {
    echo "The total number of lines: $1"
    echo "The path to the data: $2"
}

# set function to be used by bash 
export -f num_lines

# output the size and path of the dataset
wc -l ./data/yelp*.json | xargs -n 2 -I% bash -c num_lines\ %

# split datasets by years
for year in "$@"
do

    cat ./data/yelp*json | grep \"date\"\:\"$year- > ./data/reviews_$year.json
    echo "data for $year is extracted"
done

# end
exit $?

