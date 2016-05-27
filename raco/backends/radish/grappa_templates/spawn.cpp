auto {{name}} = Pipeline({{ident}}, { {{dependence_captures}} }, [=] {
{{inner_code}}
});
{{name}}.run();

