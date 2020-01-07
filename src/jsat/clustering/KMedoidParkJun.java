/*
 * Copyright (C) 2018 OpenGress
 *
 * number of cluster define before clustering is done
 */
package jsat.clustering;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.IntList;

/**
 *
 * @author OpenGress
 */
public class KMedoidParkJun implements KClusterer {

    private int num; //number of cluster
    DistanceMetric dm;
    protected int[] medoids;
    protected boolean storeMedoids = true;

    public KMedoidParkJun(int num, DistanceMetric dm) {
        this.num = num;
        this.dm = dm;
    }

    private KMedoidParkJun(KMedoidParkJun aCopy) {
        this.num = aCopy.num;
        this.dm = aCopy.dm;
    }
    
    /* Performs the actual work of KMedoidParkJun. 
     * 
     * @param data the data set to apply PAM to
     * @param medioids the array to store the indices that get chosen as the medoids. The length of the array indicates how many medoids should be obtained. 
     * @param assignments an array of the same length as <tt>data</tt>, each value indicating what cluster that point belongs to. 
     * @param parallel the value of parallel 
     * @return void
*/
    protected void cluster(DataSet data, int[] medioids, int[] assignments,boolean parallel){
    
    }
    private IntList[] createDataPartitionBasedOnCategory(ClassificationDataSet cDataSet){
        IntList [] classIndex=new IntList[cDataSet.getClassSize()];
    
    for(int i=0;i<classIndex.length;i++)
        classIndex[i]=new IntList();
    //simpan index datapoint setiap kelas
    for (int i=0;i<cDataSet.getSampleSize();i++)
        classIndex[cDataSet.getDataPointCategory(i)].add(i);
    return classIndex;
    }
    @Override
    public int[] cluster(DataSet dataSet, int clusters, boolean parallel, int[] designations) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, boolean parallel, int[] designations) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public KMedoidParkJun clone() {
        return new KMedoidParkJun(this);
    }

    @Override
    public int[] cluster(DataSet dataSet, boolean parallel, int[] designations) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    
   

    
}
