import errno
import os


class Chdir:         
  """
  Safe cd that is really a pushd then popd on leaving the scope
  """
  def __init__( self, newPath ):  
    self._newPath = newPath

  def __enter__( self ):
    self._savedPath = os.getcwd()
    os.chdir(self._newPath)

  def __exit__( self, x, y, z ):
    os.chdir( self._savedPath )

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
