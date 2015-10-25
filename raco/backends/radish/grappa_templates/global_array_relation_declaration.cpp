{% extends 'relation_declaration.cpp' %}

{% block input_relation %}
Relation<{{tuple_type}}> {{resultsym}};
{% endblock %}
