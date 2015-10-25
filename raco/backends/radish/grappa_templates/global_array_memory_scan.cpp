forall<&{{global_syncname}}>( {{inputsym}}.data, {{inputsym}}.numtuples, [=](int64_t i, {{tuple_type}}& {{tuple_name}}) {
{{inner_code}}
});
