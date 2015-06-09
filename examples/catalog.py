# Schemas corresponding to Myrial examples

{
    'public:adhoc:edges': [('src','LONG_TYPE'), ('dst', 'LONG_TYPE')],
    'public:adhoc:vertices': [('id','LONG_TYPE')],
    'public:adhoc:points': [('id','LONG_TYPE'), ('x','DOUBLE_TYPE'), ('y', 'DOUBLE_TYPE')],
    'public:adhoc:sc_points': [('id', 'LONG_TYPE'), ('v', 'DOUBLE_TYPE')],
    'public:adhoc:employee' : [('id', 'LONG_TYPE'), ('dept_id', 'LONG_TYPE'), ('name', 'STRING_TYPE'),
                                ('salary','LONG_TYPE')],
    'public:adhoc:departments' : [('id', 'LONG_TYPE'), ('name','STRING_TYPE')],
    'armbrustlab:seaflow:all_data' : [('Cruise', 'LONG_TYPE'),
                                      ('Day', 'LONG_TYPE'),
                                      ('File_Id', 'LONG_TYPE'),
                                      ('chl_small', 'DOUBLE_TYPE'),
                                      ('pe', 'DOUBLE_TYPE')],
    'public:adhoc:nodes_jstor' : [('paper_id', 'LONG_TYPE'), ('year','LONG_TYPE')],
    'public:adhoc:links_jstor' : [('p1', 'LONG_TYPE'), ('p2','LONG_TYPE')],
    'dhalperi:lineage:top_papers_jstor' : [('paper_id', 'LONG_TYPE')],
    'public:adhoc:sp2bench' : [('subject', 'STRING_TYPE'), ('predicate','STRING_TYPE'), ('object','STRING_TYPE')],
}
