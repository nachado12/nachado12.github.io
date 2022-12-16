import sys
import inspect
from pylab import *
from numpy import *

def raiseNotDefined():
  print ("Method not implemented: %s" % inspect.stack()[1][3])
  sys.exit(1)

def permute(a):
  """
  Randomly permute the elements in array a
  """
  for n in range(len(a)):
    m = int(pylab.rand() * (len(a) - n)) + n
    t = a[m]
    a[m] = a[n]
    a[n] = t

def uniq(seq, idfun=None): 
  # order preserving
  if idfun is None:
    def idfun(x): return x
  seen = {}
  result = []
  for item in seq:
    marker = idfun(item)
    # in old Python versions:
    # if seen.has_key(marker)
    # but in new ones:
    if marker in seen: continue
    seen[marker] = 1
    result.append(item)
  return result

def mode(seq):
  if len(seq) == 0:
    return 1.
  else:
    cnt = {}
    for item in seq:
      if item in cnt:
        cnt[item] += 1
      else:
        cnt[item] = 1
    maxItem = seq[0]
    for item,c in cnt.items():
      if c > cnt[maxItem]:
        maxItem = item
    return maxItem

def showTree(dt, dictionary):
    left   = dt.tree_.children_left
    right  = dt.tree_.children_right
    thresh = dt.tree_.threshold
    feats  = [ dictionary[i] for i in dt.tree_.feature ]
    value  = dt.tree_.value
    def showTree_(node, s, depth):
        for i in range(depth-1):
            print ('|    ', end='') #sys.stdout.write('|    ')
        if depth > 0:
            print ('-', end='')
            print (s, end='')
            print ('-> ', end='')
        if thresh[node] == -2: # leaf
            print ('class %d\t(%d for class 0, %d for class 1)' % (argmax(value[node]), value[node][0,0], value[node][0,1]))
        else: # internal node
            print ('%s?' % feats[node])
            showTree_(left[ node], 'N', depth+1)
            showTree_(right[node], 'Y', depth+1)

    showTree_(0, '', 0)