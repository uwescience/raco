{{comment}}
auto target = hash_tuple::hash<{{keytype}}>({{keyval}}) % Grappa::cores();
Grappa::delegate::call<async, &{{pipeline_sync}}>(target, [{{keyname}}] {
    {{inner_code}}
});