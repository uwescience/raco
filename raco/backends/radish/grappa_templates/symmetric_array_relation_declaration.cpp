{% extends 'relation_declaration.cpp' %}

{% block input_relation %}
Relation<aligned_vector<{{tuple_type}}>> {{resultsym}};
{% endblock %}
