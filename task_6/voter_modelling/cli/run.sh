#!/bin/bash

# Options can also be passed on the command line.
# These options are passed blindly to the PSL CLI.
# Ex: ./run_voter.sh -D log4j.threshold=DEBUG

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly PSL_VERSION='3.0.1-SNAPSHOT'
readonly JAR_PATH="${THIS_DIR}/psl-cli-${PSL_VERSION}.jar"
readonly RUN_SCRIPT_VERSION='2.0.2'

readonly BASE_NAME='voter-model'
readonly OUTPUT_DIRECTORY="${THIS_DIR}/inferred-votes"

readonly ADDITIONAL_PSL_OPTIONS=''

function main() {
    trap exit SIGINT

    # Fetch and prepare data
    bash "${THIS_DIR}/../data/fetchVoterData.sh"

    # Make sure we can run PSL.
    check_requirements
    check_psl_jar

    # Run PSL.
    run_psl "$@"
}

function run_psl() {
    echo "Running PSL Voter Model Inference."

    java -jar "${JAR_PATH}" \
        --config "${THIS_DIR}/${BASE_NAME}.json" \
        --output "${OUTPUT_DIRECTORY}" \
        ${ADDITIONAL_PSL_OPTIONS} "$@"

    if [[ "$?" -ne 0 ]]; then
        echo 'ERROR: Failed to run inference.'
        exit 70
    fi
}

function check_requirements() {
    type java > /dev/null 2> /dev/null
    if [[ "$?" -ne 0 ]]; then
        echo 'ERROR: java required to run project.'
        exit 13
    fi
}

function check_psl_jar() {
    if [[ ! -e "${JAR_PATH}" ]]; then
        echo "ERROR: PSL JAR file not found at ${JAR_PATH}"
        echo "Please ensure the PSL JAR file exists in the CLI directory."
        exit 1
    fi
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@" 