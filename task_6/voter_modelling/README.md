# PSL Voter Modeling

## Problem Description

Voter behavior prediction using social network influence, where voting preferences are predicted based on social relationships (friends and spouses) and demographic similarity (age). The model leverages the principle that people with closer relationships have stronger influence on each other's voting choices.

## Input Data Structure

The model uses three simple data files with tab-separated values:

### Social Relationships
```
# friends.txt
Alice    Bob      1.0
Bob      Charlie  1.0
Dave     Eve      1.0

# spouses.txt  
Dave     Eve      1.0
Frank    Grace    1.0
```

### Demographics
```
# age_similarity.txt
Alice    Bob      0.95
Charlie  Dave     0.80
Eve      Frank    0.45
```

### Known Voting (Training Data)
```
# observed_votes.txt
Alice    Democrat     1.0
Frank    Republican   1.0
Grace    Green        1.0
```
