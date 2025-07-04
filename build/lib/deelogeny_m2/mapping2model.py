import os, subprocess, sys
import math

#Ce code vient de Laurent Gueguen

alph="ACGT"

def GTR(counts, norm):
  pieq={x:sum([counts[(x,y)] for y in alph if y != x])/sum([norm[(x,y)] for y in alph if y != x]) for x in alph}
  spieq=sum(pieq.values())
  pieq={k:v/spieq for k,v in pieq.items()}

  dparam={}
  dparam["a"]=(counts[("C","T")]+counts[("T","C")])/(norm[("C","T")]+norm[("T","C")])
  dparam["b"]=(counts[("T","A")]+counts[("A","T")])/(norm[("T","A")]+norm[("A","T")])
  dparam["c"]=(counts[("T","G")]+counts[("G","T")])/(norm[("T","G")]+norm[("G","T")])
  dparam["d"]=(counts[("C","A")]+counts[("A","C")])/(norm[("C","A")]+norm[("A","C")])
  dparam["e"]=(counts[("C","G")]+counts[("G","C")])/(norm[("C","G")]+norm[("G","C")])

  dparam["theta"]=pieq["C"]+pieq["G"]
  dparam["theta1"]=pieq["A"]/(pieq["A"]+pieq["T"])
  dparam["theta2"]=pieq["G"]+(pieq["C"]+pieq["G"])

  return("GTR("+",".join([k+"="+str(v) for k,v in dparam.items()])+")")

def HKY(counts, norm):
    pieq = {x: sum([counts[(x, y)] for y in alph if y != x]) / sum([norm[(x, y)] for y in alph if y != x]) for x in alph}
    spieq = sum(pieq.values())
    pieq = {k: v / spieq for k, v in pieq.items()}

    dparam={}
    dparam['theta'] = pieq["G"] + pieq["C"] 
    dparam['theta1'] = pieq["G"] / (pieq["G"] + pieq["C"]) 
    dparam['theta2'] = pieq["A"] / (pieq["A"] + pieq["T"]) 

    transitions = (counts[("A", "G")] + counts[("G", "A")] +
                   counts[("C", "T")] + counts[("T", "C")])
    transversions = sum(counts[(x, y)] for x in alph for y in alph
                        if (x, y) not in [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")] and x != y)
    dparam['kappa'] = transitions / transversions

    return ("HKY85("+",".join([k+"="+str(v) for k,v in dparam.items()])+")")

  
def estim_seq(fseq, tree, config, output) :
  """Estimate likelihoods for all trees in ltrees on simulated sequence.
  """

  dpar={}
  dpar["FSEQ"]=fseq
  dpar["TREE"]=tree
  dpar["REP"]=output
  seq=os.path.split(fseq)[-1].split(".")[-2]
  dpar["SEQ"]=seq

  out=subprocess.run(["singularity", "exec", "--env", "PATH=/testnh/build/TestNH:$PATH", "--env", "LD_LIBRARY_PATH=/bpp-phyl/build/src:/bpp-seq/build/src:/bpp-core/build/src:$LD_LIBRARY_PATH", "/home/ubuntu/tools/bpp_debian.sif", "mapnh", f"param={config}"] + [k+"="+v for k,v in dpar.items()], stdout=subprocess.PIPE)

  ## parse files
  dcounts={}
  dnorm={}
  
  f=open(output+"/"+seq+"_counts.tsv","r")
  l=f.readline()
  while l:
    [c,v]=l.split()
    d,p = c.split("->")
    dcounts[(d,p)]=float(v)
    l=f.readline()
  f.close()

  f=open(output+"/"+seq+"_counts_norm.tsv","r")
  l=f.readline()
  while l:
    [c,v]=l.split()
    d,p = c.split("->")
    dnorm[(d,p)]=float(v)
    l=f.readline()
  f.close()

  return([dcounts, dnorm])
  

