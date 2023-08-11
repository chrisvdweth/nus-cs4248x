%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Facts
%%%%%%%%%%%%%%%%%%%%%%%%%%
human(harry).
human(ron).
human(hermione).
human(ginny).
human(dumbledore).
human(snape).
human(hagrid).
human(voldemort).
animal(hedwig).
animal(crookshanks).
animal(scabbers).
animal(trevor).
animal(buckbeak).

houseelf(dobby).
houseelf(winky).
houseelf(kreacher).

pet(hedwig).
pet(crookshanks).
pet(scabbers).
pet(trevor).
owl(hedwig).

cat(crookshanks).
rat(scabbers).
toad(trevor).
hippogriff(buckbeak).

school(hogwarts).
school(beauxbatons).
school(durmstrang).

wand(harry_wand).
wand(ron_wand_1).
wand(ron_wand_2).
wand(hermione_wand).

loves(ron, hermione).
loves(ginny, harry).

hates(snape, harry).
hates(X,Y) :- fights(X,Y).

owns(harry, harry_wand).
owns(harry, hedwig).
owns(ron, scabbers).
owns(ron, ron_wand_2).
owns(hermione, hermione_wand).
owns(hermione, crookshanks).

attends(harry, hogwarts).
attends(ron, hogwarts).
attends(hermione, hogwarts).
attends(ginny, hogwarts).

works_at(dumbledore, hogwarts).
works_at(snape, hogwarts).
works_at(hagrid, hogwarts).
works_at(dobby, hogwarts).
works_at(winky, hogwarts).

fights_with(harry, voldemort).
fights_with(dumbledore, voldemort).

%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%
fights(X,Y) :- fights_with(X,Y) ; fights_with(Y,X).
enemies(X,Y) :- fights(X,Y).

wizard(X) :- human(X), wand(Y), owns(X, Y).