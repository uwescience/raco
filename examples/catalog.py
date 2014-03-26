# Schemas corresponding to Myrial examples

{
    'public:adhoc:edges': [('src','int'), ('dst', 'int')],
    'public:adhoc:vertices': [('id','int')],
    'public:adhoc:points': [('id','int'), ('x','float'), ('y', 'float')],
    'public:adhoc:sc_points': [('v', 'float')],
    'public:adhoc:employee' : [('id', 'int'), ('dept_id', 'int'), ('name', 'string'),
                                ('salary','int')],
    'public:adhoc:departments' : [('id', 'int'), ('name','string')],
    'armbrustlab:seaflow:all_data' : [('Cruise', 'int'),
                                      ('Day', 'int'),
                                      ('File_Id', 'int'),
                                      ('chl_small', 'float'),
                                      ('pe', 'float')]
}
