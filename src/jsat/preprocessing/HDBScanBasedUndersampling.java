/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.preprocessing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import jsat.classifiers.ClassificationDataSet;
import jsat.clustering.DBSCAN;
import jsat.clustering.HDBSCAN;
import static jsat.preprocessing.ForOverlapPAMUndersampling.createDataPartitionBasedOnCategory;
import static jsat.preprocessing.PAMBasedUndersampling.getMajorityClassSample;
import static jsat.preprocessing.PAMBasedUndersampling.getMinorityClassSample;
import static jsat.preprocessing.Sampling.sumUp;
import jsat.utils.IntList;
import static jsat.utils.IntList.convertIntListToList;
import jsat.utils.random.RandomUtil;

/**
 * step:
 * 1. data diclaster menjadi sejumlah claster menggunakan HDBScan
 * 2. ambil data dari masing-masing cluster secara random sejumlah tertentu 
 *    berdasarkan persamaan: size = r * MI*(MCi/MA)
 * 3. Gabungkan data kelas + dengan data dari masing-masing claster
 * 
 */
public class HDBScanBasedUndersampling {
private   ClassificationDataSet minClass;
private ClassificationDataSet majClass;
private int sizeMI;
private int sizeMA;

public ClassificationDataSet getSampleByHBU(ClassificationDataSet cds,float r){
    //1. pisahkan data
    devideByCategory(cds);
    System.out.println("data kelas min: "+ sizeMI);
    System.out.println("data kelas min: "+ sizeMA);
    //2. lakukan clustering pada kelas mayor
    List<Integer> listIdxMA =  getClusterbasedMADataset(cds,r);
    //3. gabungkan data kelas mayor dengan kelas minor
    ClassificationDataSet newMACds = cds.getSubset(listIdxMA);
    ClassificationDataSet newCds = sumUp(minClass,newMACds);
    return newCds;
}
public ClassificationDataSet clusterDBScanSMOTE(ClassificationDataSet dataSet, double r){
     List<cluster> listC = new ArrayList();
         IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
          ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
       ClassificationDataSet majClass= getMajorityClassSample(dataSet,classIndex);
       HDBSCAN dbscan = new HDBSCAN(30);     
        int[] clusterResult= new int[majClass.getSampleSize()];
       clusterResult = dbscan.cluster(majClass, clusterResult);
       int num =numberCluster(clusterResult);
       for (int i=0;i<clusterResult.length;i++){
            cluster c = new cluster(i,clusterResult[i]);
            listC.add(c);
        }
        Collections.sort(listC, new Comparator<cluster>() {
			@Override
			public int compare(cluster c1, cluster c2) {
				return c1.getNumber()-c2.getNumber();
			}
		});
        List<List<Integer>> clusterElemen = new ArrayList();
        for(int i=0;i<num+1;i++){
            List<Integer> listkei = new ArrayList();
            clusterElemen.add(listkei);
        }
        for(int j=0;j<listC.size();j++){
            if(listC.get(j).getNumber()>=0)
            clusterElemen.get(listC.get(j).getNumber()).add(listC.get(j).getIdx());
        }
        List<Integer> newCdsIdx= new ArrayList();
        for(int k=0;k<clusterElemen.size();k++){
            List<Integer> listkek = clusterElemen.get(k);
            int sci = listkek.size();
            int sizeMACi = Math.round((float)r*(float)minClass.getSampleSize()*((float)sci/(float)majClass.getSampleSize()));
            for(int i=0;i<sizeMACi;i++){
                //int nextRand = RandomUtil.getRandom().nextInt(listkek.size()-1);
                newCdsIdx.add(listkek.get(i));
            }
            
        }
        ClassificationDataSet newcds = dataSet.getSubset(newCdsIdx);
        ClassificationDataSet cdsGab= Sampling.sumUp(minClass, newcds);
        System.out.println("cdsGab"+ cdsGab.getSampleSize());
       
        //SamplingBasedPrediction.SMOTEAndDT(cdsGab);
       return cdsGab; 
       
        
}
public ClassificationDataSet clusterDBScanSMOTERandom(ClassificationDataSet dataSet, double r){
         List<cluster> listC = new ArrayList();
         IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
          ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
       ClassificationDataSet majClass= getMajorityClassSample(dataSet,classIndex);
       HDBSCAN dbscan = new HDBSCAN(30);     
        int[] clusterResult= new int[majClass.getSampleSize()];
       clusterResult = dbscan.cluster(majClass, clusterResult);
       int num =numberCluster(clusterResult);
       for (int i=0;i<clusterResult.length;i++){
            cluster c = new cluster(i,clusterResult[i]);
            listC.add(c);
        }
        Collections.sort(listC, new Comparator<cluster>() {
			@Override
			public int compare(cluster c1, cluster c2) {
				return c1.getNumber()-c2.getNumber();
			}
		});
        List<List<Integer>> clusterElemen = new ArrayList();
        for(int i=0;i<num+1;i++){
            List<Integer> listkei = new ArrayList();
            clusterElemen.add(listkei);
        }
        for(int j=0;j<listC.size();j++){
            if(listC.get(j).getNumber()>=0)
            clusterElemen.get(listC.get(j).getNumber()).add(listC.get(j).getIdx());
        }
        List<Integer> newCdsIdx= new ArrayList();
        for(int k=0;k<clusterElemen.size();k++){
            List<Integer> listkek = clusterElemen.get(k);
            int sci = listkek.size();
            int sizeMACi = Math.round((float)r*(float)minClass.getSampleSize()*((float)sci/(float)majClass.getSampleSize()));
            for(int i=0;i<sizeMACi;i++){
                int nextRand = RandomUtil.getRandom().nextInt(listkek.size()-1);
                newCdsIdx.add(listkek.get(nextRand));
            }
            
        }
        ClassificationDataSet newcds = dataSet.getSubset(newCdsIdx);
        ClassificationDataSet cdsGab= Sampling.sumUp(minClass, newcds);
        System.out.println("cdsGab"+ cdsGab.getSampleSize());
       
        //SamplingBasedPrediction.SMOTEAndDT(cdsGab);
       return cdsGab; 
       
       
    }

public ClassificationDataSet clusterDBScan(ClassificationDataSet dataSet, double std, int mpts,double r){
         List<cluster> listC = new ArrayList();
         IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
          ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
       ClassificationDataSet majClass= getMajorityClassSample(dataSet,classIndex);
      // HDBSCAN dbscan = new HDBSCAN(30);     
        DBSCAN dbscan = new DBSCAN();
        int[] clusterResult= new int[majClass.getSampleSize()];
      // clusterResult = dbscan.cluster(majClass, clusterResult);
      dbscan.setStndDevs(std);
    clusterResult = dbscan.cluster(majClass,30,true, clusterResult);
      
       int num =numberCluster(clusterResult);
       for (int i=0;i<clusterResult.length;i++){
            cluster c = new cluster(i,clusterResult[i]);
            listC.add(c);
        }
        Collections.sort(listC, new Comparator<cluster>() {
			@Override
			public int compare(cluster c1, cluster c2) {
				return c1.getNumber()-c2.getNumber();
			}
		});
        List<List<Integer>> clusterElemen = new ArrayList();
        for(int i=0;i<num+1;i++){
            List<Integer> listkei = new ArrayList();
            clusterElemen.add(listkei);
        }
        for(int j=0;j<listC.size();j++){
            if(listC.get(j).getNumber()>=0)
            clusterElemen.get(listC.get(j).getNumber()).add(listC.get(j).getIdx());
        }
        List<Integer> newCdsIdx= new ArrayList();
        for(int k=0;k<clusterElemen.size();k++){
            List<Integer> listkek = clusterElemen.get(k);
            int sci = listkek.size();
            int sizeMACi = Math.round((float)r*(float)minClass.getSampleSize()*((float)sci/(float)majClass.getSampleSize()));
            for(int i=0;i<sizeMACi;i++){
                int nextRand = RandomUtil.getRandom().nextInt(listkek.size()-1);
                newCdsIdx.add(listkek.get(nextRand));
            }
            
        }
        ClassificationDataSet newcds = dataSet.getSubset(newCdsIdx);
        ClassificationDataSet cdsGab= Sampling.sumUp(minClass, newcds);
       // System.out.println("cdsGab"+ cdsGab.getSampleSize());
       
        //SamplingBasedPrediction.SMOTEAndDT(cdsGab);
       return cdsGab; 
       
       
    }
public int numberCluster(int[] cluster){
        int num = -1;
        for(int i=0;i<cluster.length;i++){
            if(cluster[i]>num)
                num = cluster[i];
        }
        return num;
            
        
    }
private static int idxMinClass(IntList[] sample){

        int idx =0;
        for(int i=1;i<sample.length;i++){
            if(sample[i].size()<sample[idx].size())
                idx = i;
        }
        return idx;
    }
    private static int idxMajClass(IntList[] sample){

        int idx =0;
        for(int i=1;i<sample.length;i++){
            if(sample[i].size()>sample[idx].size())
                idx = i;
        }
        return idx;
    }
public static ClassificationDataSet getMinorityClassSample(ClassificationDataSet cDataSet,IntList[] sample){
       // int min = numOfMinClassSample(sample);
       int min = idxMinClass(sample);

        ClassificationDataSet subset=null;
        return subset= cDataSet.getSubset(convertIntListToList(sample[min]));

    }

    public static ClassificationDataSet getMajorityClassSample(ClassificationDataSet cDataSet,IntList[] sample){
        int maj = idxMajClass(sample);

        ClassificationDataSet subset=null;
        return subset= cDataSet.getSubset(convertIntListToList(sample[maj]));
        //System.out.println("numNumericVal"+ subset[i].getDataPoint(0).numNumericalValues());

    }
//pisahkan data kelas positif dengan kelas negatif
    public void devideByCategory(ClassificationDataSet cds){
        IntList[] classIndex=createDataPartitionBasedOnCategory(cds);
        minClass = getMinorityClassSample(cds, classIndex);
       // System.out.println("Jumlah sampel kelas +:"+ minClass.getSampleSize());
        majClass = getMajorityClassSample(cds, classIndex);
        sizeMI = minClass.getSampleSize();
        sizeMA = majClass.getSampleSize();
    }
    
    //lakukan clustering pada data kelasn mayor untuk membagi berdasarkan claster
    public List<Integer> getClusterbasedMADataset(ClassificationDataSet cds, float r){
        devideByCategory(cds);
        HDBSCAN dbscan = new HDBSCAN(30);        
        int[] clusterResult= new int[majClass.getSampleSize()];
       // clusterResult= dbscan.cluster(dataSet, clusterResult);
       clusterResult = dbscan.cluster(majClass, true, clusterResult);
       for(int n=0;n<clusterResult.length;n++){
           System.out.println("cluster result "+ n + ": "+ clusterResult[n]);
       }
       List<List<Integer>> clusterMember = getIdxAllClusterMember(clusterResult);
       List<Integer> listOfPart = new ArrayList();
       for(int i=0;i<clusterMember.size();i++){
           int sci = clusterMember.get(i).size();
           int sizeMACi = Math.round(r*sizeMI*(sci/sizeMA));
           //ambil secara random sejumlah titik
           List<Integer> listEachCluster = getPartOfIdxFromCluster(clusterMember.get(i),sizeMACi);
           listOfPart.addAll(listEachCluster);           
       }
       return listOfPart;
    }
    
    public List<Integer> getPartOfIdxFromCluster(List<Integer> elmCluster, int num){
        List<Integer> list = new ArrayList();
        for(int i=0;i<num;i++){
            int rand = RandomUtil.getRandom().nextInt(num);
            list.add(elmCluster.get(rand));
        }
        return list;
    }
    //dari no cluster diubah menjadi list elemen cluster
    public List<List<Integer>> getIdxAllClusterMember(int[] cluster){
        int num = clusterNumber(cluster);
        System.out.println("num: "+ num);
        List<List<Integer>> list = new ArrayList();
        //menyiapkan arraylist untuk menampung
        for(int i=0;i<num;i++){
            List<Integer> listkei = new ArrayList();
            list.add(listkei);
        }
        System.out.println("cluster: "+ list.size());
        for(int j=0;j<cluster.length;j++){
            list.get(cluster[j]).add(j);
        }
        return list;
    }
    private int clusterNumber(int[] cluster){
        int max = Integer.MIN_VALUE;
        for(int i=0;i<cluster.length;i++){
            System.out.println(cluster[i]);
            if(max<cluster[i]){
                max = cluster[i];
            }
        }
        return max;
    }
     private class cluster{
        private int idx;
        private int number;
        
        public cluster(int idx,int number){
            this.idx = idx;
            this.number=number;
        }
        public int getIdx() {
            return idx;
        }

        public void setIdx(int idx) {
            this.idx = idx;
        }

        public int getNumber() {
            return number;
        }

        public void setNumber(int number) {
            this.number = number;
        }

        
        
    }
    
}
