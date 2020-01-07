/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.preprocessing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import static jsat.clustering.ClustererBase.createIdxClusterListFromAssignmentArray;
import jsat.clustering.PAM;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.DefaultVectorCollection;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionUtils;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author dell
 */
public class Sampling {
    
    public ClassificationDataSet majkmeanBasedUndersampling(ClassificationDataSet dataSet){
        KMeanBasedUndersampling kmean = new KMeanBasedUndersampling();
        
        ClassificationDataSet balanceDtset = kmean.xmeanClusterBasedUndersampling(dataSet, 1000);
        return balanceDtset;
    }
    public ClassificationDataSet majClusterBasedUndersampling(ClassificationDataSet dataSet){
        PAMBasedUndersampling pbu = new PAMBasedUndersampling();
        pbu.setEachClassMember(dataSet);
        ClassificationDataSet balanceDtset = pbu.clusterbasedUndersampling(dataSet, 100);
        return balanceDtset;
    }
    public ClassificationDataSet randomMajClusterBasedUndersampling(ClassificationDataSet dataSet){
        PAMBasedUndersampling pbu = new PAMBasedUndersampling();
       
        ClassificationDataSet balanceDtset = pbu.randomInClusterbasedUndersampling(dataSet);
        return balanceDtset;
    }
    //paper Yan and Lee 2009 : cluster based for imbalance
    public ClassificationDataSet SBCSampling(ClassificationDataSet dataSet){
        //hitung jumlah kelas minimum 
        int minTotal = this.minClassSample(dataSet).getSampleSize();
        
        //klaster seluruh data menjadi sejumlah klaster
        PAM pam = new PAM();
        int[] clusterMember = new int[dataSet.getSampleSize()] ;
        clusterMember= pam.cluster(dataSet, 3, clusterMember);
        List<List<Integer>> listcluster= createIdxClusterListFromAssignmentArray(clusterMember,dataSet);
        ClassificationDataSet c1list1 = dataSet.getSubset(listcluster.get(0));
        ClassificationDataSet c1list2 = dataSet.getSubset(listcluster.get(1));
        ClassificationDataSet c1list3 = dataSet.getSubset(listcluster.get(2));
        //untuk setiap klaster ambil kelas mayoritas sebanyak jumlah yang ditentukan
        PAMBasedUndersampling pbu1 = new PAMBasedUndersampling();
        pbu1.setEachClassMember(c1list1);
        int minsz1 = pbu1.getMinClass().getSampleSize();
        int majsz1 = pbu1.getMajClass().getSampleSize();
        PAMBasedUndersampling pbu2 = new PAMBasedUndersampling();
        pbu2.setEachClassMember(c1list2);
        int minsz2 = pbu2.getMinClass().getSampleSize();
        int majsz2 = pbu2.getMajClass().getSampleSize();
        PAMBasedUndersampling pbu3 = new PAMBasedUndersampling();
        pbu3.setEachClassMember(c1list3);
        int minsz3 = pbu3.getMinClass().getSampleSize();
        int majsz3 = pbu3.getMajClass().getSampleSize();
        
        int [] ratio = new int[3];
        ratio[0] = Math.round(majsz1/minsz1);
        ratio[1] = Math.round(majsz2/minsz2);
        ratio[2] = Math.round(majsz3/minsz3);
        
        int majSampleC1 = countMajSampleRequired(minTotal,ratio,0);
        ClassificationDataSet cds1 = pbu1.getRandomFromMajClassSample(majSampleC1);
        int majSampleC2 = countMajSampleRequired(minTotal,ratio,1);
        ClassificationDataSet cds2 = pbu2.getRandomFromMajClassSample(majSampleC2);
        int majSampleC3 = countMajSampleRequired(minTotal,ratio,2);
        ClassificationDataSet cds3 = pbu3.getRandomFromMajClassSample(majSampleC3);
        ClassificationDataSet sub1= sumUp(cds1,cds2);
        ClassificationDataSet sub2= sumUp(cds3,this.minClassSample(dataSet));
        ClassificationDataSet sub = sumUp(sub1,sub2);
        return sub;
    }
    
    public ClassificationDataSet SBCAndMajClusterSampling(ClassificationDataSet dataSet){
        //hitung jumlah kelas minimum 
        int minTotal = this.minClassSample(dataSet).getSampleSize();
        
        //klaster seluruh data menjadi sejumlah klaster
        PAM pam = new PAM();
        int[] clusterMember = new int[dataSet.getSampleSize()] ;
        clusterMember= pam.cluster(dataSet, 3, clusterMember);
        List<List<Integer>> listcluster= createIdxClusterListFromAssignmentArray(clusterMember,dataSet);
        ClassificationDataSet c1list1 = dataSet.getSubset(listcluster.get(0));
        ClassificationDataSet c1list2 = dataSet.getSubset(listcluster.get(1));
        ClassificationDataSet c1list3 = dataSet.getSubset(listcluster.get(2));
        //untuk setiap klaster ambil kelas mayoritas sebanyak jumlah yang ditentukan
        PAMBasedUndersampling pbu1 = new PAMBasedUndersampling();
        pbu1.setEachClassMember(c1list1);
        int minsz1 = pbu1.getMinClass().getSampleSize();
        int majsz1 = pbu1.getMajClass().getSampleSize();
        PAMBasedUndersampling pbu2 = new PAMBasedUndersampling();
        pbu2.setEachClassMember(c1list2);
        int minsz2 = pbu2.getMinClass().getSampleSize();
        int majsz2 = pbu2.getMajClass().getSampleSize();
        PAMBasedUndersampling pbu3 = new PAMBasedUndersampling();
        pbu3.setEachClassMember(c1list3);
        int minsz3 = pbu3.getMinClass().getSampleSize();
        int majsz3 = pbu3.getMajClass().getSampleSize();
        
        int [] ratio = new int[3];
        ratio[0] = Math.round(majsz1/minsz1);
        ratio[1] = Math.round(majsz2/minsz2);
        ratio[2] = Math.round(majsz3/minsz3);
        
        int majSampleC1 = countMajSampleRequired(minTotal,ratio,0);
        ClassificationDataSet cds1 = pbu1.getSBCbasedSample(pbu1.getMajClass(),majSampleC1);
        int majSampleC2 = countMajSampleRequired(minTotal,ratio,1);
        ClassificationDataSet cds2 = pbu2.getSBCbasedSample(pbu2.getMajClass(),majSampleC2);
        int majSampleC3 = countMajSampleRequired(minTotal,ratio,2);
        ClassificationDataSet cds3 = pbu3.getSBCbasedSample(pbu3.getMajClass(),majSampleC3);
        ClassificationDataSet sub1= sumUp(cds1,cds2);
        ClassificationDataSet sub2= sumUp(cds3,this.minClassSample(dataSet));
        ClassificationDataSet sub = sumUp(sub1,sub2);
        return sub;
    }
    private int countMajSampleRequired(int totalMin, int []r, int idx){
        int sum=0;
        for(int i=0;i<r.length;i++)
            sum+=r[i];
        return totalMin*r[idx]/(sum);
    }
    public ClassificationDataSet clusterBasedSampling(ClassificationDataSet dataSet){
    PAM pam = new PAM();
        int[] clusterMember = new int[dataSet.getSampleSize()] ;
        clusterMember= pam.cluster(dataSet, 2, clusterMember);
        
        int[] med= pam.getMedoids();
        /*
        for(int i=0;i<clusterMember.length;i++)
            System.out.print(clusterMember[i]+" ;");
            */
        List<List<Integer>> listcluster= createIdxClusterListFromAssignmentArray(clusterMember,dataSet);
             
        ClassificationDataSet c1list1 = dataSet.getSubset(listcluster.get(0));
        PAMBasedUndersampling pbu1 = new PAMBasedUndersampling();
        pbu1.setEachClassMember(c1list1);
        ClassificationDataSet c1listMaj = pbu1.getMajClass();
        int minC1 = pbu1.getMinClass().getSampleSize();
      //  System.out.println("min class: "+ pbu1.getMinClass().getSampleSize());
      //  System.out.println("maj class: "+ pbu1.getMajClass().getSampleSize());
        ClassificationDataSet balanceDtsetC1 = pbu1.clusterbasedUndersampling(c1list1, 100);
      //  System.out.println("balanceDtsetC1: "+ balanceDtsetC1.getSampleSize());
        
        PAMBasedUndersampling pbu2 = new PAMBasedUndersampling();
        ClassificationDataSet c1list2 = dataSet.getSubset(listcluster.get(1));
        pbu2.setEachClassMember(c1list2);
        ClassificationDataSet c2listMaj = pbu2.getMajClass();
        int minC2 = pbu2.getMinClass().getSampleSize();
      //  System.out.println("min class: "+ pbu2.getMinClass().getSampleSize());
      //  System.out.println("maj class: "+ pbu2.getMajClass().getSampleSize());
        ClassificationDataSet balanceDtsetC2 = pbu2.clusterbasedUndersampling(c1list2, 100);
      // System.out.println("balanceDtsetC2: "+ balanceDtsetC2.getSampleSize());
       /*
        PAMBasedUndersampling pbu3 = new PAMBasedUndersampling();
        ClassificationDataSet c1list3 = dataSet.getSubset(listcluster.get(2));
        pbu3.setEachClassMember(c1list3);
        ClassificationDataSet c3listMaj = pbu3.getMajClass();
        int minC3 = pbu3.getMinClass().getSampleSize();
       // System.out.println("min class: "+ pbu3.getMinClass().getSampleSize());
       // System.out.println("maj class: "+ pbu3.getMajClass().getSampleSize());
        
        ClassificationDataSet balanceDtsetC3 = pbu3.clusterbasedUndersampling(c1list2, 100);
       // System.out.println("balanceDtsetC3: "+ balanceDtsetC3.getSampleSize());
        
        ClassificationDataSet sub = sumUp(balanceDtsetC3,balanceDtsetC2);
        */
    ClassificationDataSet balance = sumUp(balanceDtsetC1,balanceDtsetC2);
        //suffle random 
        List<Integer> indices = new ArrayList();
        for(int i=0;i<balance.getSampleSize();i++)
            indices.add(i);
        Random random = new Random();
        Collections.shuffle(indices, random);
        ClassificationDataSet randBalance = balance.getSubset(indices);
        return balance;
    }
    public static ClassificationDataSet sumUp(ClassificationDataSet d1, ClassificationDataSet d2){
        ClassificationDataSet d = d1;
       // int classIdx = d2.getDataPoint(0)
        for(int i=0;i<d2.getSampleSize();i++){
            d.addDataPoint(d2.getDataPoint(i),d2.getDataPointCategory(i));
        }
        
        return d;
        
    }
    public ClassificationDataSet minClassSample(ClassificationDataSet dataSet){
        IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
        ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
       return minClass;
    }
    public ClassificationDataSet majClassSample(ClassificationDataSet dataSet){
        IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
        ClassificationDataSet majClass = getMajorityClassSample(dataSet, classIndex);
       return majClass;
    }
    /*
    strategi undersampling:
    1. pisahkan data kelas minoritas dari kelas mayoritas
    2. ambil bagian data kelas mayoritas sebanyak kelas minoritas
    */
    public ClassificationDataSet underSampling(ClassificationDataSet dataSet){
         IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
        ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
       // System.out.println("Jumlah sampel kelas +:"+ minClass.getSampleSize());
        ClassificationDataSet  majClass = getMajorityClassSample(dataSet, classIndex);
        Random rand= new Random();
        for(int i=0;i<minClass.getSampleSize();i++){
            int idx = rand.nextInt(minClass.getSampleSize());
            if(majClass.getSampleSize()>idx)
                minClass.addDataPoint(majClass.getDataPoint(idx),majClass.getDataPointCategory(idx));
            
        }
        return minClass;
        
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
     public  ClassificationDataSet getMinorityClassSample(ClassificationDataSet cDataSet,IntList[] sample){
        int min = idxMinClass(sample);
        
        ClassificationDataSet subset= cDataSet.getSubset(convertIntListToList(sample[min]));
        return subset;
        
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
 
    public  ClassificationDataSet SMOTESampling(ClassificationDataSet dataSet, boolean parallel, double targetRatio){
    DistanceMetric dm= new EuclideanDistance();
    int smoteNeighbors=5;
    
    
    
        if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("SMOTE only works with numeric-only feature values");
        
        ExecutorService threadPool;
        if(parallel)
            threadPool = new FakeExecutor();
        else
            threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        
        List<Vec> vAll = dataSet.getDataVectors();
        IntList[] classIndex = new IntList[dataSet.getClassSize()];
        for(int i = 0; i < classIndex.length; i++)
            classIndex[i] = new IntList();
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            classIndex[dataSet.getDataPointCategory(i)].add(i);
        
        double[] priors = dataSet.getPriors();
        Vec ratios = DenseVector.toDenseVec(priors).clone();//yes, make a copy - I want the priors around too!
        /**
         * How many samples does it take to reach parity with the majority class
         */
        final int majorityNum = (int) (dataSet.getSampleSize()*ratios.max());
        ratios.mutableDivide(ratios.max());
        
        final List<DataPointPair<Integer>> synthetics = new ArrayList<>();
        
        //Go through and perform oversampling of each class
        for(final int classID : ListUtils.range(0, dataSet.getClassSize()))
        {
           // final int samplesNeeded = (int) (majorityNum * targetRatio - classIndex[classID].size());
             final int samplesNeeded = (int) (majorityNum * targetRatio);
            System.out.println("majorityNum: "+majorityNum+ " target ratio: "+ targetRatio+" sample needed: "+ samplesNeeded);
            if(samplesNeeded <= 0)
                continue;
            //collect the vectors we need to interpolate with
            final List<Vec> V_id = new ArrayList<>();
            for(int i : classIndex[classID])
                V_id.add(vAll.get(i));
            VectorCollection<Vec> VC_id = new DefaultVectorCollection<>(dm, V_id, parallel);
            //find all the nearest neighbors for each point so we know who to interpolate with
            final List<List<? extends VecPaired<Vec, Double>>> nns_id = VectorCollectionUtils.allNearestNeighbors(VC_id, V_id, smoteNeighbors+1, threadPool);
            
            ParallelUtils.run(parallel, samplesNeeded, (start, end)->
            {
                Random rand = RandomUtil.getRandom();
                List<DataPoint> local_new = new ArrayList<>();
                for (int i = start; i < end; i++)
                {
                    int sampleIndex = i % V_id.size();
                    //which of the neighbors should we use?
                    int nn = rand.nextInt(smoteNeighbors) + 1;//index 0 is ourselve
                    VecPaired<Vec, Double> vec_nn = nns_id.get(sampleIndex).get(nn);
                    double gap = rand.nextDouble();

                    // x ~ U(0, 1)
                    //new = sample + x * diff
                    //where diff = (sample - other)
                    //equivalent to
                    //new = sample * (x+1) + other * x
                    Vec newVal = V_id.get(sampleIndex).clone();
                    newVal.mutableMultiply(gap + 1);
                    newVal.mutableAdd(gap, vec_nn);
                    local_new.add(new DataPoint(newVal));
                }

                synchronized (synthetics)
                {
                    for (DataPoint v : local_new)
                        synthetics.add(new DataPointPair<>(v, classID));
                }
            }, threadPool);
            
        }
       // ClassificationDataSet newDataSet = new ClassificationDataSet(ListUtils.mergedView(synthetics, dataSet.getAsDPPList()), dataSet.getPredicting());
        ClassificationDataSet newDataSet = new ClassificationDataSet(synthetics,dataSet.getPredicting());
       return newDataSet;
    }
}
