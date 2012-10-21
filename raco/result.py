
import pyra
import sampledb


V1 = pyra.scan("R", sampledb.__dict__)


result = pyra.select(lambda t: ((t.predicate == "knows") or (t.predicate == "holdsAccount")), V1)


pyra.dump(result)

