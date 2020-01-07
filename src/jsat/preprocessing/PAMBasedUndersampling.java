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
import java.util.Random;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.svm.Pegasos;
import jsat.classifiers.trees.DecisionTree;
import jsat.classifiers.trees.RandomForest;
import jsat.clustering.ClusterMember;
import jsat.clustering.PAM;
import jsat.clustering.SeedSelectionMethods;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.IntList;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;
import modified.NonSeedRandomBagging;

/**
 *
 * @author OpenGress
 */
public class PAMBasedUndersampling {
    int clusterNumber;
    boolean clusterIsSet;
    List<ClusterMember> clusterMember;
    List<IntList> memberList;
    ClassificationDataSet minClass;
    ClassificationDataSet majClass;
    IntList[] classIndex;

    public PAMBasedUndersampling() {
        this.clusterNumber = 0;
        this.clusterIsSet=false;
    }
    private void clustering(ClassificationDataSet dataSet){
        clusterMember= this.getDataSetClusterMember(dataSet);
        this.setMemberList();
    }

    private List<ClusterMember> getClusterMember() {
        return clusterMember;
    }

    public List<IntList> getMemberList() {
        return memberList;
    }
    
    public ClassificationDataSet getRandomDataSetCluster(ClassificationDataSet dataSet){
        clustering(dataSet);
        List<Integer> indicies = new ArrayList();
        for(int i=0;i<memberList.size();i++){
            Random r = new Random();
            indicies.add(memberList.get(i).get(r.nextInt(memberList.get(i).size())));
        }
        ClassificationDataSet newdataset=this.majClass.getSubset(indicies);
        ClassificationDataSet minclass= getMinorityClassSample(dataSet, classIndex);
        System.out.println("min class : "+minclass.getSampleSize());
        ClassificationDataSet balancedata= this.sumUp(  newdataset,minclass);
        return balancedata;
    }
    
    public ClassificationDataSet clusteringPAM(ClassificationDataSet dataSet){
        IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
        ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
        System.out.println("Jumlah sampel kelas +:"+ minClass.getSampleSize());
        ClassificationDataSet  majClass = getMajorityClassSample(dataSet, classIndex);
        
        
        // menjadikan jumlah sampel kelas minoritas sebagai jumlah klaster pada kelas mayoritas
       // int extendedSize = -100; //dibuat antara -100 sd 100
       // int numCluster = minClass.getSampleSize() + extendedSize;
        int numCluster = minClass.getSampleSize() ;
        
        PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
       // List<List<DataPoint>> clusters = pam.cluster(majClass, 1000, true);
        List<List<Integer>> clusters = pam.cluster(majClass, numCluster, true);
        int [] med= pam.getMedoids();
        List<Integer> idx = new ArrayList();
        for(int i=0;i<med.length;i++){
            idx.add(med[i]);
        }
        
        ClassificationDataSet subsetMajClass = majClass.getSubset(idx);
        ClassificationDataSet dataTraining = sumUp(subsetMajClass,minClass);
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
    private static  List<Integer> convertIntListToList(IntList listInteger){
        List<Integer> listInt= new ArrayList<Integer>();
        
        for(int i=0;i<listInteger.size();i++){
            listInt.add(listInteger.get(i));
        }
        return listInt;
    }
    
    public static  IntList[] createDataPartitionBasedOnCategory(ClassificationDataSet cDataSet){
         IntList[] classIndex = new IntList[cDataSet.getClassSize()];
    
    for(int i=0;i<classIndex.length;i++)
        classIndex[i]=new IntList();
    //simpan index datapoint setiap kelas
    for (int i=0;i<cDataSet.getSampleSize();i++)
        classIndex[cDataSet.getDataPointCategory(i)].add(i);
    return classIndex;
    }
    private void getInitialMedoid(ClassificationDataSet d, EuclideanDistance dm,final List<Double> accelCache,boolean parallel){
        List<Vec> X = d.getDataVectors();
        List<List<Double>> D = new ArrayList();
        ParallelUtils.run(parallel, X.size(), (start, end) ->
            {
                
                for (int i = start; i < end; i++){
                    List<Double> sd = new ArrayList();    
                    for(int j=start;j<end;j++)
                        {
                        double dist = dm.dist(i-1,i , X, accelCache);
                        sd.add(dist);
                        }
                    D.add(sd);
                
                }
                
            });
    
    
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
    /*
    private static  int numOfMinClassSample(IntList[] sample){
        int min = Integer.MAX_VALUE;
        for(int i=0;i<sample.length;i++)
            if(min > sample[i].size())
                min = sample[i].size();
        return min;
    }
    private static int numOfMajClassSample(IntList[] sample){
        int max = Integer.MIN_VALUE;
        for(int i=0;i<sample.length;i++)
            if(max < sample[i].size())
                max = sample[i].size();
        return max;
    }
*/
     
    public List<Integer> getMinClassIdx(ClassificationDataSet cDataSet){
        classIndex=createDataPartitionBasedOnCategory(cDataSet);
        int min = idxMinClass(classIndex);
        List<Integer> listIdx= convertIntListToList(classIndex[min]);
        return listIdx;
    }
     
    public static  ClassificationDataSet getMinorityClassSample(ClassificationDataSet cDataSet,IntList[] sample){
        int min = idxMinClass(sample);
        
        ClassificationDataSet subset=null;
        return subset= cDataSet.getSubset(convertIntListToList(sample[min]));
        
    }
     
    public static  ClassificationDataSet getMajorityClassSample(ClassificationDataSet cDataSet,IntList[] sample){
        int maj = idxMajClass(sample);
        
        ClassificationDataSet subset=null;
        return subset= cDataSet.getSubset(convertIntListToList(sample[maj]));
        //System.out.println("numNumericVal"+ subset[i].getDataPoint(0).numNumericalValues());
        
    }

    public ClassificationDataSet getMinClass() {
        return minClass;
    }

    public ClassificationDataSet getMajClass() {
        return majClass;
    }
    
    public void setMinClass(ClassificationDataSet minClass) {
        this.minClass = minClass;
    }

    public void setMajClass(ClassificationDataSet majClass) {
        this.majClass = majClass;
    }
    
    public void setEachClassMember(ClassificationDataSet dataSet){
        classIndex=createDataPartitionBasedOnCategory(dataSet);
        this.setMinClass( getMinorityClassSample(dataSet, classIndex));
       // System.out.println("Jumlah sampel kelas +:"+ minClass.getSampleSize());
        this.setMajClass(getMajorityClassSample(dataSet, classIndex));
        
    }
    
    public ClassificationDataSet getSMOTEAndSBCbasedSample(ClassificationDataSet dataSet, int clusters, double ratio){
        PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
        List<List<Integer>> clusterResult = pam.cluster(this.getMajClass(), clusters, true);
        int []medoid = pam.getMedoids();
        ClassificationDataSet cds = this.getMinClass().shallowClone();
        for(int i=0;i<medoid.length;i++){
                cds.addDataPoint(this.getMajClass().getDataPoint(medoid[i]),this.getMajClass().getDataPointCategory(medoid[i]));
            }
        //tambahkan data dari setiap anggota klaster
        Random r=new Random();
        for(int i=0;i<clusterResult.size();i++){
            int sz = clusterResult.get(i).size();
            if(sz>1){
                int idx = clusterResult.get(i).get(r.nextInt(sz));
                cds.addDataPoint(this.getMajClass().getDataPoint(idx),this.getMajClass().getDataPointCategory(idx));
            }
            
            }
        //lakukan smote
        Sampling s=new Sampling();
        ClassificationDataSet smoteMin = s.SMOTESampling(cds,true,ratio);
        
        return smoteMin;
    }
    public ClassificationDataSet getSBCbasedSample(ClassificationDataSet dataSet, int clusters){
        PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
        List<List<Integer>> clusterResult = pam.cluster(this.getMajClass(), clusters, true);
        int []medoid = pam.getMedoids();
        ClassificationDataSet cds = this.getMinClass().shallowClone();
        for(int i=0;i<medoid.length;i++){
                cds.addDataPoint(this.getMajClass().getDataPoint(medoid[i]),this.getMajClass().getDataPointCategory(medoid[i]));
            }
        return cds;
    }
    public ClassificationDataSet getRandomFromMajClassSample(int count){
        Random r= new Random();
        List<Integer> idx = new ArrayList();
        for(int i=0;i<count;i++)
            idx.add(r.nextInt(count));
        ClassificationDataSet cds = this.getMajClass().getSubset(idx);
        return cds;
    }
     public ClassificationDataSet SMOTEAndClusterbasedUndersampling(ClassificationDataSet dataSet, double ratio){
          PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
        this.setEachClassMember(dataSet);
        int numCluster = minClass.getSampleSize();
        Sampling s=new Sampling();
        ClassificationDataSet smoteMin = s.SMOTESampling(dataSet,true,ratio);
        int numcluster = smoteMin.getSampleSize();
        List<List<Integer>> clusterResult = pam.cluster(this.getMajClass(), numCluster, true);
        int []medoid = pam.getMedoids();
        for(int i=0;i<medoid.length;i++){
                smoteMin.addDataPoint(this.getMajClass().getDataPoint(medoid[i]),this.getMajClass().getDataPointCategory(medoid[i]));
            }
            return smoteMin;
        
     }
     public ClassificationDataSet clusterbasedUndersampling(ClassificationDataSet dataSet, int extra){
        int numCluster = minClass.getSampleSize();
            System.out.println("numCluster: "+ numCluster);
            int numextra=0;
         if(Math.abs(extra) < Math.round(numCluster*0.01) ){
            numextra = extra;
         }
         
         PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
        if(numCluster != 0){
           // List<List<Integer>> clusterResult = pam.cluster(this.getMajClass(), numCluster+numextra, true);
            List<List<Integer>> clusterResult = pam.cluster(this.getMajClass(), numCluster, true);
            int []medoid = pam.getMedoids();
           // System.out.println("medoid: "+ medoid.length);
            ClassificationDataSet min = this.getMinClass();
           // System.out.println("min dataset: "+ min.getSampleSize());
            for(int i=0;i<medoid.length;i++){
                min.addDataPoint(this.getMajClass().getDataPoint(medoid[i]),this.getMajClass().getDataPointCategory(medoid[i]));
            }
            return min;
        }else{
            return this.getMajClass();
        }
        
        
         
         
        
        
    }
     public ClassificationDataSet randomInClusterbasedUndersampling(ClassificationDataSet dataSet){
        this.setEachClassMember(dataSet);
         PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
        List<List<Integer>> clusterResult = pam.cluster(this.getMajClass(), this.getMinClass().getSampleSize(), true);
        ClassificationDataSet min = this.getMinClass();
        Random r=new Random();
        for(int i=0;i<clusterResult.size();i++){
            int sz = clusterResult.get(i).size();
            if(sz>1){
                int idx = clusterResult.get(i).get(r.nextInt(sz));
                min.addDataPoint(this.getMajClass().getDataPoint(idx),this.getMajClass().getDataPointCategory(idx));
            }
            
            }
        return min;
        
    }
    private List<ClusterMember> getDataSetClusterMember(ClassificationDataSet dataSet){
        classIndex=createDataPartitionBasedOnCategory(dataSet);
        this.setMinClass( getMinorityClassSample(dataSet, classIndex));
       // System.out.println("Jumlah sampel kelas +:"+ minClass.getSampleSize());
        this.setMajClass(getMajorityClassSample(dataSet, classIndex));
        
        
        // menjadikan jumlah sampel kelas minoritas sebagai jumlah klaster pada kelas mayoritas
       // int extendedSize = -100; //dibuat antara -100 sd 100
        //int numCluster = minClass.getSampleSize() + extendedSize;
        int numCluster = minClass.getSampleSize();
        PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
        int[]clusterResult = new int[majClass.getSampleSize()];
        clusterResult= pam.cluster(majClass, clusterResult);
        this.clusterNumber = pam.getMedoids().length;
        this.clusterIsSet=true;
        
        
        List<ClusterMember> listC = new ArrayList();
        for (int i=0;i<clusterResult.length;i++){
            ClusterMember c = new ClusterMember(i,clusterResult[i]);
            listC.add(c);
        }
        Collections.sort(listC, new Comparator<ClusterMember>() {
			@Override
			public int compare(ClusterMember c1, ClusterMember c2) {
				return c1.getNumber()-c2.getNumber();
			}
		});
        return listC;
        
    }
    
    public IntList getClusterMember( int number){
        IntList list = new IntList();
        for(ClusterMember cm : this.getClusterMember()){
            if(cm.getNumber()==number)
                list.add(cm.getIdx());
        }
        return list;
        
    }

    public int getClusterNumber() {
        return clusterNumber;
    }
    private void setMemberList(){
        memberList = new ArrayList<>();
        for(int i = 0; i < this.clusterMember.size(); i++)
        {
            while(memberList.size() <= clusterMember.get(i).getNumber())
                memberList.add(new IntList());
            if(clusterMember.get(i).getNumber() >= 0)
                memberList.get(clusterMember.get(i).getNumber()).add(clusterMember.get(i).getIdx());
        }
        
    }
    
    
}
