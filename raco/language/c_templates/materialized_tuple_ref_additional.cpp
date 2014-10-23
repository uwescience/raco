public:
    static {{tupletypename}} fromRelationInfo(relationInfo * rel, int row) {
      {{tupletypename}} _t;
      for (int i=0; i<{{numfields}}; i++) {
         _t._fields[i] = *(void**)(&(rel->relation[row*rel->fields+i]));
         }
      return _t;
    }
