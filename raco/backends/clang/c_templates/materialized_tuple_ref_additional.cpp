public:
    static {{tupletypename}} fromRelationInfo(relationInfo * rel, int row) {
        // DOESN'T WORK WITH SCHEMAS WITH STRINGS
      {{tupletypename}} _t;
      {% for ft in fieldtypes %}
         _t.f{{loop.index-1}} = *({{ft}}*)(&(rel->relation[row*rel->fields+{{loop.index-1}}]));
      {% endfor %}
      return _t;
    }
