          // can be just the necessary schema
  class {{tupletypename}} {
    // Invariant: data stored in _fields is always in the representation
    // specified by _scheme.

    public:
    static std::tuple<{{ fieldtypes | join(',') }}> _scheme;

    void * _fields[{{numfields}}];

    template <int field>
    typename std::tuple_element<field,decltype(_scheme)>::type get() const {
      return *(typename std::tuple_element<field,decltype(_scheme)>::type *)&_fields[field];
    }

    template <int field, typename T>
    void set(T val) {
      // first be sure to convert to proper type, based on scheme
      typename std::tuple_element<field,decltype(_scheme)>::type __val = val;
      std::memcpy(&_fields[field], &__val, sizeof(int64_t));
    }

    static constexpr int numFields() {
      return {{numfields}};
    }

    {{tupletypename}} () {
      // no-op
    }

    {{tupletypename}} (const decltype(_fields)& data) {
      std::memcpy(&_fields, &data, sizeof(_fields));
    }

    // shamelessly terrible disambiguation: one solution is named factory methods
    {{tupletypename}} (std::vector<int64_t> vals, bool ignore1, bool ignore2) {
      std::memcpy(&_fields, &vals[0], sizeof(_fields));
    }

    // use the tuple schema to interpret the input stream
    static {{tupletypename}} fromIStream(std::istream& ss) {
        decltype({{tupletypename}}::_scheme) _t;

        ss
        {% for i in range(numfields) %}
            >> std::get<{{i}}>(_t)
        {% endfor %}
        ;

        {{tupletypename}} _ret;
        TupleUtils::assign<0, decltype(_scheme)>(_ret._fields, _t);
        return _ret;
    }

    void toOStream(std::ostream& os) const {
       for (int i=0; i<numFields(); i++) {
         os.write((char *)&_fields[i], sizeof(int64_t));
       }
    }

    // note not typesafe!!
    template <typename T1, typename T2>
    static {{tupletypename}} create(const T1& t1, const T2& t2) {
      //TODO: format of assertion for type safe memcpy; fo this for each field
      //static_assert(!(std::is_integral<typename std::tuple_element<field,decltype(_fields)>::type>::value ^ std::is_integral<T>::value), "Type mismatch");

        static_assert({{tupletypename}}::numFields() == (T1::numFields() + T2::numFields()), "lhs and rhs must have equal number of fields");
        {{tupletypename}} t;
        std::memcpy(&(t._fields), &(t1._fields), T1::numFields()*sizeof(int64_t));
        std::memcpy(((char*)&(t._fields))+T1::numFields()*sizeof(int64_t), &(t2._fields), T2::numFields()*sizeof(int64_t));
        return t;
    }

    template <typename T>
    static {{tupletypename}} create(const T& from) {
      static_assert({{tupletypename}}::numFields() == T::numFields(), "constructor only works on same num fields");
      {{tupletypename}} t;
      std::memcpy(&(t._fields), &(from._fields), from.numFields()*sizeof(int64_t));
      return t;
    }

    template <typename Tuple, typename T>
    {{tupletypename}} (const Tuple& v0, const T& from) {
        constexpr size_t v0_size = std::tuple_size<Tuple>::value;
        constexpr int from_size = T::numFields();
        static_assert({{tupletypename}}::numFields() == (v0_size + from_size), "constructor only works on same number of total fields");
        TupleUtils::assign<0, decltype(_scheme)>(_fields, v0);
        std::memcpy(((char*)&_fields)+v0_size*sizeof(int64_t), &(from._fields), from_size*sizeof(int64_t));
    }

    template <typename Tuple>
    {{tupletypename}} (const Tuple& v0) {
        static_assert({{tupletypename}}::numFields() == (std::tuple_size<Tuple>::value), "constructor only works on same number of total fields");
        TupleUtils::assign<0, decltype(_scheme)>(_fields, v0);
    }

    std::ostream& dump(std::ostream& o) const {
      o << "Materialized(";

      // for (int i=0; i<numFields(); i++) {
      //  o << _fields[i] << ",";
      // }
      TupleUtils::str(o, (void**)_fields, _scheme);

      o << ")";
      return o;
    }

    {{additional_code}}
  } {{after_def_code}};

  std::ostream& operator<< (std::ostream& o, const {{tupletypename}}& t) {
    return t.dump(o);
  }

  std::tuple<{{ fieldtypes | join(',') }}> {{tupletypename}}::_scheme;

