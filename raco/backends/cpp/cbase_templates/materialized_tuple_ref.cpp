          // can be just the necessary schema
  class {{tupletypename}} {
    // Invariant: data stored in _fields is always in the representation
    // specified by _scheme.

    public:
    {% for ft in fieldtypes %}
        {{ft}} f{{loop.index - 1}};
    {% endfor %}

    static constexpr int numFields() {
      return {{numfields}};
    }

    // size of all fields in struct removing only end padding
    static size_t fieldsSize() {
        const {{tupletypename}} _t;
        return

        {% if numfields == 0 %}
            0;
        {% elif numfields == 1 %}
            sizeof(_t.f0);
         {% else %}
            ((char*)&_t.f{{numfields-1}}) + sizeof(_t.f{{numfields-1}}) - ((char*)&_t);
         {% endif %}
    }

    // debugging function to get storage sizes
    static void print_representation() {
        const {{tupletypename}} _t;

        {% if numfields == 0 %}
            // noop
        {% elif numfields == 1 %}
        std::cout << _t.fieldsSize() << std::endl;
        {% else %}
        {% for i in range(1, numfields) %}
        std::cout << (((char*)&_t.f{{i}}) - ((char*)&_t.f{{i-1}})) << ",";
        {% endfor %}
        std::cout << (_t.fieldsSize() - (((char*)&_t.f{{numfields-2}}) - ((char*)&_t)));
        std::cout << std::endl;
        {% endif %}


    }

    {{tupletypename}} () {
      // no-op
    }

    //template <typename OT>
    //{{tupletypename}} (const OT& other) {
    //  std::memcpy(this, &other, sizeof({{tupletypename}}));
    //}
    {% if numfields > 0 %}
    {{tupletypename}} ({% for ft in fieldtypes %}
                               const {{ft}}& a{{loop.index-1}}
                               {% if not loop.last %},{% endif %}
                       {% endfor %}
                       ) {
        {% for i in range(numfields) %}
            f{{i}} = a{{i}};
        {% endfor %}
    }

    {# list of types comma separated #}
    {# TODO: uncomment when jinja 2.8 is released with support for set blocks
    {% set types_comma %}
        {% for ft in fieldtypes %}
        {{ft}}
        {% if not loop.last %},{% endif %}
        {% endfor %}
    {% endset %}
    #}
    {% endif %}


    {{tupletypename}}(const std::tuple<
        {% for ft in fieldtypes %}
        {{ft}}
        {% if not loop.last %},{% endif %}
        {% endfor %}
            >& o) {
        {% for i in range(numfields) %}
            f{{i}} = std::get<{{i}}>(o);
        {% endfor %}
     }

     std::tuple<
        {% for ft in fieldtypes %}
        {{ft}}
        {% if not loop.last %},{% endif %}
        {% endfor %}
     > to_tuple() {

        std::tuple<
        {% for ft in fieldtypes %}
        {{ft}}
        {% if not loop.last %},{% endif %}
        {% endfor %}
        > r;
        {% for i in range(numfields) %}
            std::get<{{i}}>(r) = f{{i}};
        {% endfor %}
        return r;
     }

    // shamelessly terrible disambiguation: one solution is named factory methods
    //{{tupletypename}} (std::vector<int64_t> vals, bool ignore1, bool ignore2) {
    //    {% for i in range(numfields) %}
    //        f{{i}} = vals[{{i}}];
    //    {% endfor %}
    //}

    // use the tuple schema to interpret the input stream
    static {{tupletypename}} fromIStream(std::istream& ss, char delim=' ') {
        {{tupletypename}} _ret;

        {% for i in range(numfields) %}
            {% if fieldtypes[i] == string_type_name %}
               {
               std::string _temp;
               std::getline(ss, _temp, delim);
               // beware; truncate= true, so unckecked truncation for stringlen >= MAX_STR_LEN!
               _ret.f{{i}} = to_array<MAX_STR_LEN, std::string, true>(_temp);
               }
            {% else %}
               {
               // use operator>> to parse into proper numeric type
               ss >> _ret.f{{i}};
               //throw away the next delimiter
               std::string _temp;
               std::getline(ss, _temp, delim);
               }
            {% endif %}
        {% endfor %}

        return _ret;
    }

    void toOStream(std::ostream& os) const {
        os.write((char*)this, this->fieldsSize());
    }

    void toOStreamAscii(std::ostream& os) const {
        os
        {% if numfields == 0 %}
           << "" <<
        {% else %}
        {% for i in range(numfields-1) %}
        << f{{i}} << " "
        {% endfor %}
        << f{{numfields-1}} <<
        {% endif %}
        std::endl;
    }

    //template <typename Tuple, typename T>
    //{{tupletypename}} (const Tuple& v0, const T& from) {
    //    constexpr size_t v0_size = std::tuple_size<Tuple>::value;
    //    constexpr int from_size = T::numFields();
    //    static_assert({{tupletypename}}::numFields() == (v0_size + from_size), "constructor only works on same number of total fields");
    //    TupleUtils::assign<0, decltype(_scheme)>(_fields, v0);
    //    std::memcpy(((char*)&_fields)+v0_size*sizeof(int64_t), &(from._fields), from_size*sizeof(int64_t));
    //}

    //template <typename Tuple>
    //{{tupletypename}} (const Tuple& v0) {
    //    static_assert({{tupletypename}}::numFields() == (std::tuple_size<Tuple>::value), "constructor only works on same number of total fields");
    //    TupleUtils::assign<0, decltype(_scheme)>(_fields, v0);
    //}

    std::ostream& dump(std::ostream& o) const {
      o << "Materialized(";

      {% for i in range(numfields) %}
        o << f{{i}} << ",";
      {% endfor %}

      o << ")";
      return o;
    }

    {{additional_code}}
  } {{after_def_code}};

  std::ostream& operator<< (std::ostream& o, const {{tupletypename}}& t) {
    return t.dump(o);
  }

