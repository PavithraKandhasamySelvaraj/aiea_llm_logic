% facts

male(ross).
male(joey).
male(chandler).
male(mike).

female(monica).
female(rachel).
female(phoebe).

friend(joey, chandler).
friend(chandler, joey).
friend(monica, rachel).
friend(monica, phoebe).

husband(mike, phoebe).
husband(chandler, monica).

sibling(ross, monica).
sibling(monica, ross).

% Rules

bestfriend(X, Y) :- friend(X, Y), friend(Y, X).
wife(X, Y) :- female(X), male(Y), husband(Y, X).
brother(X, Y) :- male(X), sibling(X, Y).
sister(X, Y) :- female(X), sibling(X, Y).


