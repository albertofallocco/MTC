#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from random import shuffle, sample


class MTClust():
    

    def __repr__(self):
        
        return "<Metric Partitioning Cluster Object v.4>"
    
    
    def __init__(self, max_depth=0, min_instances=0, dist_function="euclidean"):
        
        self.max_level=max_depth
        self.min_instances=max(2,min_instances)
        
        if dist_function=="euclidean": 
            self.distFunction=self.distEuclidean
        elif dist_function=="manhattan": 
            self.distFunction=self.distManhattan
        elif dist_function=="jaccard": 
            self.distFunction=self.distJaccard
        else:
            print("Distance not recognized. Switching to Euclidean.")
            self.distFunction=self.distEuclidean
       
    
    def distEuclidean(self, data, tree, slist):
        
        dat=data[slist]
        
        LEFT  = self.data[tree[0][0]]
        RIGHT = self.data[tree[1][0]]
        K0=(np.dot(RIGHT,RIGHT)-np.dot(LEFT,LEFT))*.5
        DELTA=RIGHT-LEFT
        
        r=np.dot(dat, DELTA) < K0
        
        nodes_left  = slist[r]
        nodes_right = slist[~r]
        
        return nodes_left, nodes_right

            
    def distManhattan(self, data, tree, slist):
        r=np.sum(np.abs(data[slist]-self.data[tree[0][0]]), axis=1) < np.sum(np.abs(data[slist]-self.data[tree[1][0]]), axis=1)
        nodes_left  = slist[r]
        nodes_right = slist[~r]
        return nodes_left, nodes_right
    
    
    '''PREVIOUS IMPLEMENTATION - does not work (& operator not supported
    def distJaccard(self, data, tree, slist):
    
        a=self.data[tree[0][0]]
        b=self.data[tree[1][0]]
        Aa=len(a)
        Ab=len(b)
        res=[]
        for ix in slist:
            item=data[ix]
            B=len(item)
            Ca=len(a & item)
            Cb=len(b & item)
            res.append(Ca/(Aa+B-Ca) > Cb/(Ab+B-Cb))
            
        r=np.array(res, dtype=bool)
        nodes_left  = slist[r]
        nodes_right = slist[~r]
        
        return nodes_left, nodes_right
        return nodes_left, nodes_right
       '''
    
    def distJaccard(self, data, tree, slist):
        #NEW IMPLEMENTATION - works, but poor results

        a = self.data[tree[0][0]]
        b = self.data[tree[1][0]]

        size_a = np.sum(a)
        size_b = np.sum(b)

        jaccard_sim_a = np.sum(np.minimum(data[slist], a), axis=1) / np.sum(np.maximum(data[slist], a), axis=1)
        jaccard_sim_b = np.sum(np.minimum(data[slist], b), axis=1) / np.sum(np.maximum(data[slist], b), axis=1)

        res = jaccard_sim_a > jaccard_sim_b

        nodes_left = slist[res]
        nodes_right = slist[~res]

        return nodes_left, nodes_right

    
    def addToTree(self, slist, tree, clusters, level=0, path=""):

        if self.max_level>0 and level>= self.max_level: 
            for item in slist: clusters[item]=path
            return

        nodes_left, nodes_right= self.distFunction(self.data, tree, slist)

        if len(nodes_left)>=self.min_instances:
            ltree=tree[0][1]
            
            ltree.append((nodes_left[0], []))
            ltree.append((nodes_left[1], []))

            nodes_left=np.concatenate((nodes_left[2:], nodes_left[:2]))
            self.addToTree(nodes_left, ltree, clusters, level+1, path+"L")
            
        else: 
            for item in nodes_left: clusters[item]=path+"L"

        if len(nodes_right)>=self.min_instances:
            rtree=tree[1][1]
            
            rtree.append((nodes_right[0], []))
            rtree.append((nodes_right[1], []))
            
            nodes_right=np.concatenate((nodes_right[2:], nodes_right[:2]))
            self.addToTree(nodes_right, rtree, clusters, level+1, path+"R")
        else:
            for item in nodes_right: clusters[item]=path+"R"
            
            
    def travelTree(self, items, data, tree, clusters, path="", level = 0):
        
        if self.max_level > 0 and level >= self.max_level: 
            for item in items: clusters[item]=path
            return
        if len(tree)<2: 
            for item in items: clusters[item]=path
            return
        
        nodes_left, nodes_right = self.distFunction(data, tree, items)
        if len(nodes_left)>0:
            self.travelTree(nodes_left,  data, tree[0][1], clusters, path+"L", level+1)
        if len(nodes_right)>0:
            self.travelTree(nodes_right, data, tree[1][1], clusters, path+"R", level+1)


    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.N=data.values.shape[0]
            self.data=data.values
        elif isinstance(data, np.ndarray):
            self.N=data.shape[0]
            self.data=data
        else:
            raise Exception("only pandas dataframes or numpy arrays, please")

        L=list(range(self.N))
        shuffle(L)
        L=np.array(L)
      
        clusters=[-1]*self.N  
        self.tree=[(L[-1], []), (L[-2], [])]
      
        self.addToTree(L, self.tree, clusters)
     
        cs=set(clusters)
       
        self.clusters=dict(zip(cs, range(len(cs))))
       
        self.data_clusts=np.array([self.clusters[c] for c in clusters])
        
        return self
        
    def predict(self, data):
        if isinstance(data, pd.DataFrame) :
            data_N=data.values.shape[0]
            ndata=data.values
        elif isinstance(data, np.ndarray):
            data_N=data.shape[0]
            ndata=data
        else:
            raise Exception("only pandas dataframes or numpy arrays, please")
        
        L=np.array(list(range(data_N)))
        clusters=[-1]*data_N
        self.travelTree(L, ndata, self.tree, clusters)
        clusts=[self.clusters[cl] for cl in clusters]
        return np.array(clusts)

    
class MTClustT(MTClust):
   
    def __repr__(self):
        return "<Metric Partitioning Cluster Object (target oriented) v.3>"

    def distEuclidean(self, data, tree, list_a, list_b=None):

        LEFT  = self.data[tree[0][0]]
        RIGHT = self.data[tree[1][0]]
        K0=(np.dot(RIGHT,RIGHT)-np.dot(LEFT,LEFT))*.5
        DELTA=RIGHT-LEFT
                  
        if list_b is None:
            
            dat_a=data[list_a]
            r=np.dot(dat_a, DELTA) < K0

            nodes_left  = list_a[r]
            nodes_right = list_a[~r]
            
            return nodes_left, nodes_right
        else: 
            
            dat_a=data[list_a]
            dat_b=data[list_b]
            rA=np.dot(dat_a, DELTA) < K0
            rB=np.dot(dat_b, DELTA) < K0
            nodes_left_a  = list_a[ rA]
            nodes_right_a = list_a[~rA]
            nodes_left_b  = list_b[ rB]
            nodes_right_b = list_b[~rB]
            return nodes_left_a, nodes_right_a, nodes_left_b, nodes_right_b

    def distManhattan(self, data, tree, list_a, list_b=None):
        if list_b is None:
           
            r=np.sum(np.abs(data[list_a]-self.data[tree[0][0]]), axis=1) < np.sum(np.abs(data[list_a]-self.data[tree[1][0]]), axis=1)
            nodes_left  = list_a[ r]
            nodes_right = list_a[~r]
            return nodes_left, nodes_right
        else: 
           
            rA = np.sum(np.abs(data[list_a]-self.data[tree[0][0]]), axis=1) < np.sum(np.abs(data[list_a]-self.data[tree[1][0]]), axis=1)
            rB = np.sum(np.abs(data[list_b]-self.data[tree[0][0]]), axis=1) < np.sum(np.abs(data[list_b]-self.data[tree[1][0]]), axis=1)
            nodes_left_a  = list_a[ rA]
            nodes_right_a = list_a[~rA]
            nodes_left_b  = list_b[ rB]
            nodes_right_b = list_b[~rB]
            return nodes_left_a, nodes_right_a, nodes_left_b, nodes_right_b
    
    
    '''SAME AS MTCLUST
    def jaccard_aux(self, data, the_list, a, b, Aa, Ab):
        res=[]
        for ix in the_list:
            item=data[ix]
            B=len(item)
            Ca=len(a & item)
            Cb=len(b & item)
            res.append(Ca/(Aa+B-Ca) > Cb/(Ab+B-Cb))
        
        #?
        r = [item for item, condition in zip(the_list, res) if condition]
        nodes_left = r
        nodes_right = [item for item, condition in zip(the_list, res) if not condition]
        #?
            
        #r=np.array(res, dtype=bool)
        #nodes_left  = the_list[r]
        #nodes_right = the_list[~r]
        
        return nodes_left, nodes_right
        
        
    def distJaccard(self, data, tree, list_a, list_b=None):
        a=self.data[tree[0][0]]
        b=self.data[tree[1][0]]
        Aa=len(a)
        Ab=len(b)
        if list_b is None:
            left, right= self.jaccard_aux(data, list_a, a, b, Aa, Ab)
            return left, right
        else:
            left_a, right_a= self.jaccard_aux(data, list_a, a, b, Aa, Ab)
            left_b, right_b= self.jaccard_aux(data, list_b, a, b, Aa, Ab)
            return left_a, right_a, left_b, right_b'''
    
    
    def jaccard_aux(self, data, the_list, a, b):
        #NEW IMPLEMENTATION - again, poor results
        Aa = np.sum(a)
        Ab = np.sum(b)

        jaccard_sim_a = np.sum(np.minimum(data[the_list], a), axis=1) / np.sum(np.maximum(data[the_list], a), axis=1)
        jaccard_sim_b = np.sum(np.minimum(data[the_list], b), axis=1) / np.sum(np.maximum(data[the_list], b), axis=1)

        res = jaccard_sim_a > jaccard_sim_b

        nodes_left = [item for item, condition in zip(the_list, res) if condition]
        nodes_right = [item for item, condition in zip(the_list, res) if not condition]

        
        return nodes_left, nodes_right


    def distJaccard(self, data, tree, list_a, list_b=None):

        a = self.data[tree[0][0]]
        b = self.data[tree[1][0]]

        if list_b is None:
            left, right = self.jaccard_aux(data, list_a, a, b)
            return left, right

        else:
            left_a, right_a = self.jaccard_aux(data, list_a, a, b)
            left_b, right_b = self.jaccard_aux(data, list_b, a, b)
            return left_a, right_a, left_b, right_b


    def addToTree(self, list_a, list_b, tree, level=0):

        if self.max_level>0 and level>= self.max_level: return
        if len(list_a)==0 or len(list_b)==0: return
        
        nodes_left_a, nodes_right_a, nodes_left_b, nodes_right_b = self.distFunction(self.data, tree, list_a, list_b) 
        if len(nodes_left_a)>0 and len(nodes_left_b)>0:
            ltree=tree[0][1]
            ltree.append((nodes_left_a[0], []))
            ltree.append((nodes_left_b[0], []))
            self.addToTree(nodes_left_a[1:],nodes_left_b[1:], ltree, level+1)
        if len(nodes_right_a)>0 and len(nodes_right_b)>0:
            rtree=tree[1][1]
            rtree.append((nodes_right_a[0], []))
            rtree.append((nodes_right_b[0], []))
            self.addToTree(nodes_right_a[1:],nodes_right_b[1:], rtree, level+1)
            

    def fit(self, data_X, data_y):
        if isinstance(data_X, pd.DataFrame):
            self.N=data_X.values.shape[0]
            self.data=data_X.values
        elif isinstance(data_X, np.ndarray):
            self.N=data_X.shape[0]
            self.data=data_X
        else:
            raise Exception("only pandas dataframes or numpy arrays, please")
        
        self.data_y=data_y
        the_cls=list(set(self.data_y))
        if len(the_cls)!=2: raise Exception("Sorry, we can only do binary classifiers at this moment. Please stay tuned for a new version!")
       
        L=np.array(range(self.N))
        List_A =L[self.data_y==the_cls[0]]
        List_B =L[self.data_y==the_cls[1]]
        shuffle(List_A)
        shuffle(List_B)
        List_A=np.array(List_A)
        List_B=np.array(List_B)
        self.tree=[(List_A[0], []), (List_B[0], [])]
        self.addToTree(List_A[1:], List_B[1:], self.tree)
      
        L=np.array(list(range(self.N)))
        clusters=[-1]*self.N
        self.travelTree(L, self.data, self.tree, clusters)
        #print(clusters)
        cs=set(clusters)
        self.clusters=dict(zip(cs, range(len(cs))))
        self.data_clusts=np.array([self.clusters[c] for c in clusters])
        return self
        
class MTClassifier():
    
    def __init__(self, n_parts=5, max_depth=0, min_instances=0, use_binary_class=False, dist_func="euclidean"):
        self.n_parts = n_parts
        self.min_instances=min_instances 
        self.max_depth = max_depth
        self.binary_class = use_binary_class
        self.dist_func=dist_func
        
    def eval_simple(self, mtc):
      
        Nc=len(mtc.clusters)
      
        class_counter=np.array([[0]*len(self.y_classes) for i in range(Nc)])
        
        for i, yi in enumerate(self.truth_i): class_counter[mtc.data_clusts[i]][yi]+=1
        
        best = np.argmax(class_counter, axis=1)
        return np.array(best)

    def eval_prob(self, mtc):

        Nc=len(mtc.clusters)
        class_counter=np.array([[0]*len(self.y_classes) for i in range(Nc)])
        
        for i, yi in enumerate(self.truth_i): class_counter[mtc.data_clusts[i]][yi]+=1
        class_counter = class_counter/class_counter.sum(axis=1, keepdims=True)
        #print(class_counter)
        return class_counter

    def fit(self,X, y, verbose=False):
        self.parts=[]
        self.eval_results =[]
        self.eval_resultsP=[]
      
        self.y_classes=sorted(list(set(y)))
        yclass_index=dict(zip(self.y_classes, range(len(self.y_classes))))
        self.truth_i=[yclass_index[ay] for ay in y]
        
        #print(yclass_index)
        if self.binary_class==True: print("Binary classification")
        for i in range(self.n_parts):
            if verbose==True: print("%3d partition" % (i+1))
            if self.binary_class==False: 
                mc = MTClust(max_depth=self.max_depth, min_instances=self.min_instances, dist_function=self.dist_func)
                mc.fit(X)
            else:
                mc = MTClustT(max_depth=self.max_depth, min_instances=self.min_instances, dist_function=self.dist_func)
                #mc = MTClustT(self.max_depth, self.dist_func)
                mc.fit(X, y)
            self.parts.append(mc)
            #we will probably only need one of these
            res_simple=self.eval_simple(mc)   
            res_probs=self.eval_prob( mc)     
            #ditto
            self.eval_results.append(res_simple)
            self.eval_resultsP.append(res_probs)     
        return self
    
    def predict_basic(self, Xt):
        for i in range(self.n_parts):
            mc=self.parts[i]
          
            pclusters=mc.predict(Xt)
 
            r=self.eval_results[i][pclusters]
            if i==0:
                res_mat=r
            else:
                res_mat=np.vstack([res_mat,r])
     
        res_mat=res_mat.T
        N=res_mat.shape[0]
        if res_mat.dtype in (np.dtype('int32'), np.dtype('int64')) :
            if self.n_parts>1:
                results=[np.argmax(np.bincount(res_mat[i])) for i in range(N)]
            else:
                results=[res_mat[i] for i in range(N)]
        else:
            results=[]
            tvals= set(self.truth)
        
            for row in res_mat:
                unique, counts = np.unique(row, return_counts=True)
                r=sorted(zip(counts, unique), reverse=True)[0][1]
              
                results.append(r)
        
        results=[self.y_classes[i] for i in results]
        return np.array(results)
    

    def predict_abs(self, Xt):
        
        for i in range(self.n_parts):
            mc=self.parts[i]
     
            pclusters=mc.predict(Xt)
          
            r=self.eval_resultsP[i][pclusters]
            if i==0:
                res_mat=r
            else:
               
                res_mat+=r
        results=np.argmax(res_mat, axis=1)
        results=[self.y_classes[i] for i in results]

        return results

    def predict(self, Xt, epsilon=0.01):
        
        NClasses=len(self.y_classes)
        Ps = np.ones((Xt.shape[0], NClasses))
        
        e=NClasses*epsilon      
        eps=np.ones(len(self.y_classes))*e

        for i in range(self.n_parts):
            mc=self.parts[i]
            
            pclusters=mc.predict(Xt)
           
            r=self.eval_resultsP[i][pclusters] + eps
            Ps *= r
        results=np.argmax(Ps, axis=1)
        results=[self.y_classes[i] for i in results]

        return results


    def predict_proba(self, Xt, return_dict=False, epsilon=0.01):
 
        NClasses=len(self.y_classes)
        Ps = np.ones((Xt.shape[0], NClasses))
        
        e=NClasses*epsilon        
        eps=np.ones(len(self.y_classes))*e

        for i in range(self.n_parts):
            mc=self.parts[i]
        
            pclusters=mc.predict(Xt)
          
            r=self.eval_resultsP[i][pclusters] + eps
            Ps *= r
        
        sums=np.sum(Ps, axis=1)
        probs=Ps.T/sums
        probs=probs.T
        
        if return_dict == False:
            return probs
        else:
            results=[]
            for plist in probs:
                results.append({mc1.y_classes[c]: v for c, v in enumerate(plist)})
            return results


# In[ ]:




