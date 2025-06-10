female(monica).
female(rachel).
female(phoebe).
female(amy).
male(ross).
male(joey).
male(chandler).
male(mike).
friend(joey, chandler).
friend(chandler, joey).
friend(monica, rachel).
friend(monica, phoebe).
husband(mike, phoebe).
husband(chandler, monica).
sister(X, Y) :- female(X), sibling(X, Y).
sibling(amy, rachel).