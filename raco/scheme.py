from raco import expression
import raco.types

from collections import OrderedDict


class DummyScheme(object):
    """Dummy scheme used to generate plans in the absence of catalog info."""
    def __len__(self):
        return 0

    def __repr__(self):
        return "DummyScheme()"


class Scheme(object):
    """Add an attribute to the scheme."""
    salt = "1"

    def __init__(self, attributes=None):
        if attributes is None:
            attributes = []
        self.attributes = []
        self.asdict = OrderedDict()
        for n, t in attributes:
            self.addAttribute(n, t)

    def addAttribute(self, name, _type):
        assert _type in raco.types.ALL_TYPES, \
            'Invalid type name: %s' % str(_type)
        _type = raco.types.map_type(_type)

        if name in self.asdict:
            # ugly.  I don't like throwing errors in this case, but it's worse
            # not to
            return self.addAttribute(name + self.salt, _type)
        self.asdict[name] = (len(self.attributes), _type)
        self.attributes.append((name, _type))
        # just in case we changed the name.  ugly.
        return name

    def get_types(self):
        """Return a list of the types in this scheme."""
        return [_type for name, _type in self.attributes]

    def get_names(self):
        """Return a list of the names in this scheme."""
        return [name for name, _type in self.attributes]

    def typecheck(self, tup):
        rmap = raco.types.reverse_python_type_map
        try:
            return all([rmap[_type](v) for (_, _type), v in
                       zip(self.attributes, tup)])
        except:
            raise TypeError("%s not of type %s" % (tup, self.attributes))

    def __eq__(self, other):
        return self.attributes == other.attributes

    def __ne__(self, other):
        return not (self == other)

    def getPosition(self, name):
        return self.asdict[name][0]

    def getName(self, position):
        return self[position][0]

    def getType(self, name):
        if type(name) == int:
            return self[name][1]
        else:
            return self.asdict[name][1]

    def resolve(self, attrref):
        """return the name and type of the attribute reference, resolved
        against this scheme"""
        unnamed = expression.toUnnamed(attrref, self)
        return self.getName(unnamed.position), self.getType(unnamed.position)

    def ascolumnlist(self):
        """Return a columnlist structure suitable for use with Project and
        ProjectingJoin. Currently a list of positional attribute references.
        May eventually be a scheme itself."""
        return [expression.UnnamedAttributeRef(i) for i in xrange(len(self))]

    def __contains__(self, attr, typ=None):
        if typ:
            return (attr, type) in self.attributes
        else:
            return attr in self.asdict

    def __str__(self):
        """Pretty print the scheme"""
        return str(self.attributes)

    def __repr__(self):
        return "Scheme({att!r})".format(att=self.attributes)

    def __len__(self):
        """Return the number of attributes in the scheme"""
        return len(self.attributes)

    def __iter__(self):
        return self.attributes.__iter__()

    def __getitem__(self, key):
        return self.attributes.__getitem__(key)

    def __add__(self, other):
        newsch = Scheme(self.attributes)
        for (n, t) in other:
            newsch.addAttribute(n, t)
        return newsch


class EmptyScheme(Scheme):
    def __init__(self):
        Scheme.__init__(self, [])
