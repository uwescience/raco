emp = scan(public:adhoc:employee);
dept = scan(public:adhoc:departments);
out = select emp.name as emp_name, dept.name as dept_name
      from dept, emp
      where emp.dept_id == dept.id AND emp.salary > 5000;
store(out, OUTPUT);
