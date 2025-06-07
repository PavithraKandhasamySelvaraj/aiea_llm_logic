#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly BASE_NAME='voter'
readonly DATA_DIR="${THIS_DIR}/${BASE_NAME}"

PEOPLE=(Alice Bob Charlie Dave Eve Frank Grace)
PARTIES=(Democrat Republican Green)

function main() {
    trap exit SIGINT

    if [[ -e "${DATA_DIR}" ]]; then
        echo "Data directory already exists. To fetch fresh data, remove the directory: ${DATA_DIR}"
        return
    fi

    mkdir -p "${DATA_DIR}/0/learn"
    mkdir -p "${DATA_DIR}/0/eval"

    # Create learn data
    create_learn_data

    # Create eval data
    create_eval_data

    echo "Data fetched and prepared in: ${DATA_DIR}"
}

function create_learn_data() {
    # Friends data
    cat > "${DATA_DIR}/0/learn/friends_obs.txt" << EOF
Alice	Bob	1.0
Bob	Charlie	1.0
Dave	Eve	1.0
EOF

    # Spouses data
    cat > "${DATA_DIR}/0/learn/spouses_obs.txt" << EOF
Dave	Eve	1.0
Frank	Grace	1.0
EOF

    # Age similarity data
    cat > "${DATA_DIR}/0/learn/age_similarity_obs.txt" << EOF
Alice	Bob	0.95
Charlie	Dave	0.80
Eve	Frank	0.45
EOF

    # Known votes (observations) - only include known votes
    cat > "${DATA_DIR}/0/learn/votes_obs.txt" << EOF
Alice	Democrat	1.0
Frank	Republican	1.0
Grace	Green	1.0
EOF

    # Vote targets (what we want to predict) - all (person, party) pairs except those in votes_obs.txt
    > "${DATA_DIR}/0/learn/votes_targets.txt"
    for person in "${PEOPLE[@]}"; do
        for party in "${PARTIES[@]}"; do
            if ! grep -q -E "^${person}[[:space:]]+${party}[[:space:]]" "${DATA_DIR}/0/learn/votes_obs.txt"; then
                echo -e "${person}	${party}" >> "${DATA_DIR}/0/learn/votes_targets.txt"
            fi
        done
    done

    # Ground truth for evaluation - include all votes
    cat > "${DATA_DIR}/0/learn/votes_truth.txt" << EOF
Alice	Democrat	1.0
Alice	Republican	0.0
Alice	Green	0.0
Bob	Democrat	0.85
Bob	Republican	0.10
Bob	Green	0.05
Charlie	Democrat	0.45
Charlie	Republican	0.35
Charlie	Green	0.20
Dave	Democrat	0.20
Dave	Republican	0.75
Dave	Green	0.05
Eve	Democrat	0.20
Eve	Republican	0.75
Eve	Green	0.05
Frank	Democrat	0.0
Frank	Republican	1.0
Frank	Green	0.0
Grace	Democrat	0.0
Grace	Republican	0.0
Grace	Green	1.0
EOF
}

function create_eval_data() {
    # Copy the same data for evaluation
    cp "${DATA_DIR}/0/learn/friends_obs.txt" "${DATA_DIR}/0/eval/friends_obs.txt"
    cp "${DATA_DIR}/0/learn/spouses_obs.txt" "${DATA_DIR}/0/eval/spouses_obs.txt"
    cp "${DATA_DIR}/0/learn/age_similarity_obs.txt" "${DATA_DIR}/0/eval/age_similarity_obs.txt"
    cp "${DATA_DIR}/0/learn/votes_obs.txt" "${DATA_DIR}/0/eval/votes_obs.txt"
    cp "${DATA_DIR}/0/learn/votes_truth.txt" "${DATA_DIR}/0/eval/votes_truth.txt"

    # Vote targets for eval (same logic as learn)
    > "${DATA_DIR}/0/eval/votes_targets.txt"
    for person in "${PEOPLE[@]}"; do
        for party in "${PARTIES[@]}"; do
            if ! grep -q -E "^${person}[[:space:]]+${party}[[:space:]]" "${DATA_DIR}/0/eval/votes_obs.txt"; then
                echo -e "${person}	${party}" >> "${DATA_DIR}/0/eval/votes_targets.txt"
            fi
        done
    done
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@" 