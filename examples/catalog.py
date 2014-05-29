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
                                      ('pe', 'DOUBLE_TYPE')]
}
