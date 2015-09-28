{{comment}}
auto target = hash_tuple::hash<{{keytype}}>()({{keyval}}) % Grappa::cores();
// DEV NOTE: if something inside this call is not captured in the lambda,
// (probably a data structure) then we need to change its declaration to a global one.
// The alternative is just to capture [=] but this will mask unneeded communication.
Grappa::delegate::call<async, &{{pipeline_sync}}>(target, [{{keyname}}] {
    {{inner_code}}
});