auto {{hashname}} = {{initializer}};
on_all_cores([=] {
   *({{hashname}}.localize()) = {{func_name}}_init();
});
