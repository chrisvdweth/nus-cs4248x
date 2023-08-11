import re
from itertools import product
from nltk.tree import Tree




def load_grammar(file_name, comment='#', default_prob=0.0):
    
    with open(file_name) as file:
        
        N, T, R = set(), set(), set()
        
        for line in file:
            line = line.strip()
            
            # Ignore lines that are comments
            if line.startswith(comment):
                continue
                
            try:
                # Split rule into left and right-hand side
                rule = line.split('->')
                
                # Extract left and right-hand side
                lhs, rhs = rule[0].strip(), rule[1].strip()
                
                # Split all right-hand sides
                rhs_parts = [ p.strip() for p in rhs.split('|') ]
                
                # Add left hand-side (must be a non-terminal) to N
                N.add(lhs)
                
                for p in rhs_parts:
                    if '[' not in p and ']' not in p:
                        prob = default_prob
                        target = tuple([ s.strip() for s in p.split(' ') if s.strip() != '' ])
                    else:
                        # Extract probability
                        m = re.search(r"\[([0-9.]+)\]", p)
                        prob = float(m.group(1))
                        target = tuple([ s.strip() for s in p.split(' ') if s.strip() != '' and '[' not in s ])

                    # Identify all terminal symbols (= symbols in quotes)
                    for t in target:
                        if t.startswith("'") and t.endswith("'"):
                            T.add(target[0][1:-1])

                    R.add((lhs, tuple([ t.replace("'", "") for t in target ]), prob))
                
                
            except Exception as e:
                print(e)
                print(line)
                continue

    return list(N), list(T), list(R)



def create_nltk_tree(tree):
    tree_str = str(tree)
    tree_str = re.sub(',', '', tree_str)
    tree_str = re.sub("'", '', tree_str)
    return Tree.fromstring(tree_str)



def reconstruct_parse_trees(PTR, i, j, sym):
    """
    Reconstruct the parse tree
    """
    trees = []
    if len(PTR[i][j][sym]) == 1 and isinstance(PTR[i][j][sym][0], str): # terminals
        return [(sym, PTR[i][j][sym][0])]
    else:
        for p1, p2 in PTR[i][j][sym]:
            i,k,B = p1
            k,j,C = p2
            for left_tree, right_tree in product(reconstruct_parse_trees(PTR, i, k, B),
                                                 reconstruct_parse_trees(PTR, k, j, C)):
                trees.append((sym, left_tree, right_tree))
        return trees