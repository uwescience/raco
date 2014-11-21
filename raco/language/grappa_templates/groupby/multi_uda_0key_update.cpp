auto {{hashname}}_local_ptr = {{hashname}}.localize();
*{{hashname}}_local_ptr = {{update_func}}(*{{hashname}}_local_ptr, {{update_val}});
