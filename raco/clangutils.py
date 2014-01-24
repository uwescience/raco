
# TODO:
# The following is actually a staged materialized tuple ref.
# we should also add a staged reference tuple ref that just has relationsymbol and row  
class StagedTupleRef:
  nextid = 0
  
  @classmethod
  def genname(cls):
    x = cls.nextid
    cls.nextid+=1
    return "t_%03d" % x
  
  def __init__(self, relsym, scheme):
    self.name = self.genname()
    self.relsym = relsym
    self.scheme = scheme
    self.__typename = None
  
  def getTupleTypename(self):
    if self.__typename==None:
      fields = ""
      relsym = self.relsym
      for i in range(0, len(self.scheme)):
        fieldnum = i
        fields += "_%(fieldnum)s" % locals()
        
      self.__typename = "MaterializedTupleRef_%(relsym)s%(fields)s" % locals()
    
    return self.__typename

    
  def generateDefition(self):
    fielddeftemplate = """int64_t _fields[%(numfields)s];
    """
    copytemplate = """_fields[%(fieldnum)s] = rel->relation[row*rel->fields + %(fieldnum)s];
    """
    template = """
          // can be just the necessary schema
  class %(tupletypename)s {
    private:
    %(fielddefs)s
    
    public:
    int64_t get(int field) const {
      return _fields[field];
    }
    
    void set(int field, int64_t val) {
      _fields[field] = val;
    }
    
    int numFields() const {
      return %(numfields)s;
    }
    
    %(tupletypename)s (relationInfo * rel, int row) {
      %(copies)s
    }
    
    %(tupletypename)s () {
      // no-op
    }

    %(tupletypename)s (std::vector<int64_t> vals) {
      for (int i=0; i<vals.size(); i++) _fields[i] = vals[i];
    }
    
    std::ostream& dump(std::ostream& o) const {
      o << "Materialized(";
      for (int i=0; i<numFields(); i++) {
        o << _fields[i] << ",";
      }
      o << ")";
      return o;
    }
    
    %(additional_code)s
  };
  std::ostream& operator<< (std::ostream& o, const %(tupletypename)s& t) {
    return t.dump(o);
  }

  """
    getcases = ""
    setcases = ""
    copies = ""
    numfields = len(self.scheme)
    fielddefs = fielddeftemplate % locals()

    
    additional_code = self.__additionalDefinitionCode__()

    # TODO: actually list the trimmed schema offsets
    for i in range(0, numfields):
      fieldnum = i
      copies += copytemplate % locals()

    tupletypename = self.getTupleTypename()
    relsym = self.relsym
      
    code = template % locals()
    return code
  
  def __additionalDefinitionCode__(self):
    return ""
  
  
  