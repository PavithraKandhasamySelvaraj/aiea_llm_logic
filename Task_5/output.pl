friend(joey, chandler).
friend(chandler, joey).
best_friend(joey, chandler).

married(mike, phoebe).
male(mike).
female(phoebe).
wife(X, Y) :- married(Y, X), female(X).

sibling(ross, monica).
sibling(monica, ross).
male(ross).
brother(X, Y) :- sibling(X, Y), male(X).

friend(monica, rachel).
friends_with(X, Y) :- friend(X, Y).