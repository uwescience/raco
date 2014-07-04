from emitcode import emitCode
from raco.language.grappalang import GrappaAlgebra
from raco.language.clang import CCAlgebra

import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

tr = "sp2bench_1m"
queries = {}

queries['Q1'] = """A(yr) :- %(tr)s(journal, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Journal'),
                 %(tr)s(journal, 'http://purl.org/dc/elements/1.1/title', 'Journal 1 (1940)'),
                 %(tr)s(journal, 'http://purl.org/dc/terms/issued', yr)"""

#"""A(inproc, author, booktitle, title, proc, ee, page, url, yr, abstract) :- %(tr)s(inproc, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Inproceedings'),
queries['Q2'] = """A(inproc, author, booktitle, title, proc, ee, page, url, yr) :- %(tr)s(inproc, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Inproceedings'),
   %(tr)s(inproc, 'http://purl.org/dc/elements/1.1/creator', author),
   %(tr)s(inproc, 'http://localhost/vocabulary/bench/booktitle', booktitle),
   %(tr)s(inproc, 'http://purl.org/dc/elements/1.1/title', title),
   %(tr)s(inproc, 'http://purl.org/dc/terms/partOf', proc),
   %(tr)s(inproc, 'http://www.w3.org/2000/01/rdf-schema#seeAlso', ee),
   %(tr)s(inproc, 'http://swrc.ontoware.org/ontology#pages',  page),
   %(tr)s(inproc, 'http://xmlns.com/foaf/0.1/homepage', url),
   %(tr)s(inproc, 'http://purl.org/dc/terms/issued', yr)"""
# TODO: make abstract optional (can do this with a union)
# TODO: order by yr

queries['Q3a'] = """A(article) :- %(tr)s(article, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
                       %(tr)s(article, property, value),
                       property = 'http://swrc.ontoware.org/ontology#pages'"""
queries['Q3b'] = """A(article) :- %(tr)s(article, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
                       %(tr)s(article, property, value),
                       property = 'http://swrc.ontoware.org/ontology#month'"""
queries['Q3c'] = """A(article) :- %(tr)s(article, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
                       %(tr)s(article, property, value),
                       property = 'http://swrc.ontoware.org/ontology#isbn'"""

#queries['Q4'] = """A(name1, name2) :- %(tr)s(article1, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
queries['Q4'] = """A(name1, name2) :- %(tr)s(article1, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
                           %(tr)s(article2, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
                           %(tr)s(article1, 'http://purl.org/dc/elements/1.1/creator', author1),
                           %(tr)s(author1, 'http://xmlns.com/foaf/0.1/name', name1),
                           %(tr)s(article2, 'http://purl.org/dc/elements/1.1/creator', author2),
                           %(tr)s(author2, 'http://xmlns.com/foaf/0.1/name', name2),
                           %(tr)s(article1, 'http://swrc.ontoware.org/ontology#journal', journal),
                           %(tr)s(article2, 'http://swrc.ontoware.org/ontology#journal', journal)"""
# TODO: name1<name2 condition (not supported
# TODO be sure DISTINCT


# syntactically join with equality;
#queries['Q5a'] = """A(person, name) :- %(tr)s(article, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
queries['Q5a'] = """A(person, name) :- %(tr)s(article, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
                            %(tr)s(article, 'http://purl.org/dc/elements/1.1/creator', person),
                            %(tr)s(inproc, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Inproceedings'),
                            %(tr)s(inproc, 'http://purl.org/dc/elements/1.1/creator', person2),
                            %(tr)s(person, 'http://xmlns.com/foaf/0.1/name', name),
                            %(tr)s(person2, 'http://xmlns.com/foaf/0.1/name', name2),
                            name = name2"""
# syntactically join with naming
#TODO: include q5b after issue #104 is addressed
#queries['Q5b'] = """A(person, name) :- %(tr)s(article, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
queries['Q5b'] = """A(person, name) :- %(tr)s(article, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Article'),
                            %(tr)s(article, 'http://purl.org/dc/elements/1.1/creator', person),
                            %(tr)s(inproc, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://localhost/vocabulary/bench/Inproceedings'),
                            %(tr)s(inproc, 'http://purl.org/dc/elements/1.1/creator', person),
                            %(tr)s(person, 'http://xmlns.com/foaf/0.1/name', name)"""

# TODO: Q6 requires negation

# TODO: Q7 requires double negation


#TODO: enable Q8, after dealing with HashJoin( $0 != $7 ) type of cases
#queries['Q8'] = """Erdoes(erdoes) :- %(tr)s(erdoes, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/Person'),
_ = """Erdoes(erdoes) :- %(tr)s(erdoes, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/Person'),
                          %(tr)s(erdoes, 'http://xmlns.com/foaf/0.1/name', "Paul Erdoes")
        A(name) :- Erdoes(erdoes),
                   %(tr)s(doc, 'http://purl.org/dc/elements/1.1/creator', erdoes),
                   %(tr)s(doc, 'http://purl.org/dc/elements/1.1/creator', author),
                   %(tr)s(doc2, 'http://purl.org/dc/elements/1.1/creator', author),
                   %(tr)s(doc2, 'http://purl.org/dc/elements/1.1/creator', author2),
                   %(tr)s(author2, 'http://xmlns.com/foaf/0.1/name', name),
                   author != erdoes,
                   doc2 != doc,
                   author2 != erdoes,
                   author2 != author

         A(name) :- Erdoes(erdoes),
                    %(tr)s(doc, 'http://purl.org/dc/elements/1.1/creator', erdoes),
                    %(tr)s(doc, 'http://purl.org/dc/elements/1.1/creator', author),
                    %(tr)s(author, 'http://xmlns.com/foaf/0.1/name', name),
                    author != erdoes"""
#TODO be sure DISTINCT

queries['Q9'] = """A(predicate) :- %(tr)s(person, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/Person'),
                        %(tr)s(subject, predicate, person)

        A(predicate) :- %(tr)s(person, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/Person'),
                        %(tr)s(person, predicate, object)"""
#TODO be sure DISTINT


queries['Q10'] = """A(subj, pred) :- %(tr)s(subj, pred, 'http://localhost/persons/Paul_Erdoes')"""

queries['Q11'] = """A(ee) :- %(tr)s(publication, 'http://www.w3.org/2000/01/rdf-schema#seeAlso', ee)"""
#TODO order by, limit, offset

alg = CCAlgebra
prefix=""
import sys
if len(sys.argv) > 1:
    if sys.argv[1] ==  "grappa" or sys.argv[1] == "g":
        print "using grappa"
        alg = GrappaAlgebra
        prefix="grappa"

plan = None
if len(sys.argv) > 2:
    plan = sys.argv[2]

for name, query in queries.iteritems():
    query = query % locals()
    lst = []
    if prefix: lst.append(prefix)
    if plan: lst.append(plan)
    if name: lst.append(name)
    emitCode(query, "_".join(lst), alg, plan)

