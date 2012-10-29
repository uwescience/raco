
class Scheme:
  '''
Add an attribute to the scheme
Type is a function that returns true for any value that is of the correct type
  '''
  salt = "1"
  def __init__(self,attributes=[]):
    self.attributes = []
    self.asdict = {}
    for n,t in attributes:
      self.addAttribute(n,t)

  def addAttribute(self, name, type):
    if self.asdict.has_key(name):
      # ugly.  I don't like throwing errors in this case, but it's worse not to
      return self.addAttribute(name + self.salt, type)
    self.asdict[name] = (len(self.attributes), type)
    self.attributes.append((name,type))
    # just in case we changed the name.  ugly.
    return name

  def typecheck(self, tup):
    try:
      return all([tf(v) for (n,tf),v in zip(self.attributes,tup)])
    except:
      raise TypeError("%s not of type %s" % (tup,self.attributes))

  def __eq__(self, other):
    return self.attributes == other.attributes

  def getPosition(self, name): 
    return self.asdict[name][0]
  
  def getType(self, name):
    return self.asdict[name][1]

  def subScheme(self, attributes):
    return Scheme([(n,self.getType(n)) for n in attributes])

  def contains(self, names):
    return all([n in self.asdict.keys() for n in names])

  def rename(self, name1, name2):
    try:
      i,t = self.asdict.pop(name1)
      self.attributes[i] = (name2, t)
      self.asdict[name2] = (i,t)
    except KeyError:
      pass

  def __iter__(self):
    return self.attributes.__iter__()

  def __getitem__(self, key):
    return self.attributes.__getitem__(key)

  def __add__(self, other):
    newsch = Scheme(self.attributes)
    for (n,t) in other:
      newsch.addAttribute(n,t)
    return newsch

class EmptyScheme(Scheme):
  def __init__(self):
    Scheme.__init__(self, [])

