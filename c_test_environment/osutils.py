import errno
import os


class Chdir:         
  """
  Safe cd that is really a pushd then popd on leaving the scope
  """
  def __init__( self, newPath ):  
    self.savedPath = os.getcwd()
    os.chdir(newPath)

  def __del__( self ):
    os.chdir( self.savedPath )

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
