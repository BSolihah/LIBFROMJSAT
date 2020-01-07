/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.preprocessing;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.kmeans.XMeans;
import jsat.linear.Vec;
import jsat.utils.IntList;
import jsat.utils.random.RandomUtil;

/**
 *data latih diclaster terlebih dahulu
 * ditambahkan parameter clusterBound pada masing-masing klaster
 * Setiap klaster dibangun cluster based classifier
 * sebuah data uji akan di tentukan dulu klasternya sebelum dipilih klassifiernya
 * 
 */
public class KMeanBasedUndersampling {
    
    public ClassificationDataSet xmeanClusterBasedUndersampling(ClassificationDataSet dataSet,int clusters){
        IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
        ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
        System.out.println("Jumlah sampel kelas +:"+ minClass.getSampleSize());
        ClassificationDataSet  majClass = getMajorityClassSample(dataSet, classIndex);
        CategoricalData categorie = dataSet.getPredicting();
     
        XMeans xmean = new XMeans();
        xmean.setMinClusterSize(clusters);
        
        List<List<Integer>> clusterResult= xmean.cluster(majClass, minClass.getSampleSize(), true);
        
        ClassificationDataSet dataTraining = minClass.shallowClone();
        List<Vec> mean= xmean.getMeans();
        System.out.println(mean.size()+","+mean.get(0).length());
        
       for(int i=0;i<mean.size();i++){
           dataTraining.addDataPoint(mean.get(i),majClass.getDataPointCategory(0));
       }
            
        System.out.println("Jumlah sampel total:"+ dataTraining.getSampleSize());
        
       return dataTraining;
    }
    
    private  ClassificationDataSet sumUp(ClassificationDataSet d1, ClassificationDataSet d2){
        ClassificationDataSet d = d1;
       // int classIdx = d2.getDataPoint(0)
        for(int i=0;i<d2.getSampleSize();i++){
            d.addDataPoint(d2.getDataPoint(i),d2.getDataPointCategory(i));
        }
        return d;
        
    }
    
    private  List<Integer> convertIntListToList(IntList listInteger){
        List<Integer> listInt= new ArrayList<Integer>();
        
        for(int i=0;i<listInteger.size();i++){
            listInt.add(listInteger.get(i));
        }
        return listInt;
    }
    private  int numOfMinClassSample(IntList[] sample){
        int min = Integer.MAX_VALUE;
        for(int i=0;i<sample.length;i++)
            if(min > sample[i].size())
                min = sample[i].size();
        return min;
    }
    private int numOfMajClassSample(IntList[] sample){
        int max = Integer.MIN_VALUE;
        for(int i=0;i<sample.length;i++)
            if(max < sample[i].size())
                max = sample[i].size();
        return max;
    }
    
     private  ClassificationDataSet getMinorityClassSample(ClassificationDataSet cDataSet,IntList[] sample){
        int min = numOfMinClassSample(sample);
        int idxMin=-1;
        for(int i=0;i<sample.length;i++)
            if(sample[i].size()==min)
                idxMin=i;
        ClassificationDataSet subset=null;
        return subset= cDataSet.getSubset(convertIntListToList(sample[idxMin]));
        
    }
    
    private  ClassificationDataSet getMajorityClassSample(ClassificationDataSet cDataSet,IntList[] sample){
        int maj = numOfMajClassSample(sample);
        int idxMaj=-1;
        for(int i=0;i<sample.length;i++)
            if(sample[i].size()==maj)
                idxMaj=i;
        ClassificationDataSet subset=null;
        return subset= cDataSet.getSubset(convertIntListToList(sample[idxMaj]));
        //System.out.println("numNumericVal"+ subset[i].getDataPoint(0).numNumericalValues());
        
    }
    
     private  IntList[] createDataPartitionBasedOnCategory(ClassificationDataSet cDataSet){
         IntList[] classIndex = new IntList[cDataSet.getClassSize()];
    
    for(int i=0;i<classIndex.length;i++)
        classIndex[i]=new IntList();
    //simpan index datapoint setiap kelas
    for (int i=0;i<cDataSet.getSampleSize();i++)
        classIndex[cDataSet.getDataPointCategory(i)].add(i);
    return classIndex;
    }
}
