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
friend(ross, rachel).
friend(ross, monica).
friend(ross, chandler).
friend(ross, joey).
friend(rachel, monica).
friend(chandler, monica).
friend(chandler, ross).
friend(jerry, phoebe).
husband(mike, phoebe).
husband(chandler, monica).

sister(X, Y) :- female(X), sibling(X, Y).
sibling(X, Y) :- sibling(Y, X).