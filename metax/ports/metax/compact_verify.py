"""Exact full-vocabulary node argmax, path-prefix selection and logits-row gather."""
import torch,triton
import triton.language as tl
@triton.jit
def _argmax(L,P,S:tl.constexpr,V:tl.constexpr,B:tl.constexpr):
 n=tl.program_id(0);i=tl.arange(0,B)
 x=tl.load(L+n*S+i,i<V,other=-float('inf'))
 tl.store(P+n,tl.argmax(x,0,tie_break_left=True).to(tl.int64))
@triton.jit
def _accept(P,R,C,O,NN:tl.constexpr,NP:tl.constexpr,NS:tl.constexpr,RS:tl.constexpr,CS:tl.constexpr,BP:tl.constexpr,BS:tl.constexpr,offset:tl.constexpr,threshold):
 p=tl.arange(0,BP);s=tl.arange(0,BS)
 node=tl.load(R+p[:,None]*RS+s[None,:],(p[:,None]<NP)&(s[None,:]<NS-1),other=0)
 node=tl.where(node<0,node+NN,node)
 pred=tl.load(P+node)+offset
 candidate=tl.load(C+p[:,None]*CS+s[None,:]+1,(p[:,None]<NP)&(s[None,:]<NS-1),other=-100000)
 ok=(tl.abs(pred-candidate)<=threshold)&(s[None,:]<NS-1)
 lengths=tl.sum(tl.cumprod(ok.to(tl.int32),1),1)
 lengths=tl.where(p<NP,lengths,-1)
 best=tl.argmax(lengths,0,tie_break_left=True);length=tl.max(lengths,0)
 node_best=tl.load(R+best*RS+length)
 node_best=tl.where(node_best<0,node_best+NN,node_best)
 bonus=tl.load(P+node_best)+offset
 tl.store(O,best.to(tl.int64));tl.store(O+1,length.to(tl.int64));tl.store(O+2,bonus.to(tl.int64));tl.store(O+3,node_best.to(tl.int64))
@triton.jit
def _row(L,O,Y,S:tl.constexpr,V:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);node=tl.load(O+3)
 x=tl.load(L+node*S+i,i<V,other=0);tl.store(Y+i,x,i<V)
def compact_accept(logits,retrieve,candidates,threshold,offset=0):
 assert logits.shape[0]==1 and logits.stride(-1)==1
 nodes,vocab=logits.shape[1:];paths,sequence=candidates.shape
 predicted=torch.empty(nodes,device=logits.device,dtype=torch.int64)
 control=torch.empty(4,device=logits.device,dtype=torch.int64)
 sample=torch.empty(vocab,device=logits.device,dtype=logits.dtype)
 _argmax[(nodes,)](logits,predicted,logits.stride(1),vocab,triton.next_power_of_2(vocab),num_warps=8)
 _accept[(1,)](predicted,retrieve,candidates,control,nodes,paths,sequence,retrieve.stride(0),candidates.stride(0),triton.next_power_of_2(paths),triton.next_power_of_2(sequence-1),int(offset),0.0 if threshold is None else float(threshold),num_warps=4)
 _row[(triton.cdiv(vocab,1024),)](logits,control,sample,logits.stride(1),vocab,1024)
 return control[0],control[1],sample,control[2]
