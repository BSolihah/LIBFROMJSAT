/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifier.clusterbased;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.clustering.kmeans.KMeans;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.preprocessing.PAMBasedUndersampling;

/**
 *
 * @author dell
 */
public class KMeanCusBagDT implements Classifier, Parameterized{

    @Parameter.ParameterHolder
    protected Classifier classifier1;
    protected Classifier classifier2;
    protected PAMBasedUndersampling pam;
    protected KMeans kmean;
    public KMeanCusBagDT(Classifier classifier1,Classifier classifier2 ){
      this.classifier1 = classifier1;
      this.classifier2 = classifier2;
      this.kmean = new HamerlyKMeans();
      this.pam = new PAMBasedUndersampling();
    }
    
    
    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel) {
        //bagi dataset kedalam dua klaster
        
        int[] clusterResult= new int[dataSet.getSampleSize()];
        clusterResult= kmean.cluster(dataSet, 2, clusterResult);
        List<cluster> listC = new ArrayList();
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
        List<cluster> cluster1= new ArrayList();
        List<cluster> cluster2= new ArrayList();
        
        int idx =0;
        while(idx < listC.size()){            
            int n = listC.get(idx).getNumber();
            switch (n){
                case 0:
                    cluster1.add(listC.get(idx));
                    break;
                case 1:
                    cluster2.add(listC.get(idx));
                    break;
                
            }
            idx +=1;
            
        }
        ClassificationDataSet dataSetC1 = getDataSetOfCluster(cluster1,dataSet);
        ClassificationDataSet dataSetC2 = getDataSetOfCluster(cluster2,dataSet);
        //lakukan undersampling pada masing-masing dataSet
        ClassificationDataSet dataTrainC1 = pam.clusteringPAM(dataSetC1);
        ClassificationDataSet dataTrainC2 = pam.clusteringPAM(dataSetC2);
        //lakukan training pada masing masing cluster
        this.classifier1.train(dataTrainC1,parallel);
        this.classifier2.train(dataTrainC2,parallel);
        
    }
    
    private  ClassificationDataSet getDataSetOfCluster(List<cluster> list, ClassificationDataSet dataSet){
        List<Integer> listIdx = new ArrayList();
        for(cluster c: list){
            listIdx.add(c.getIdx());
        }
        ClassificationDataSet dataCluster = dataSet.getSubset(listIdx);
        return dataCluster;
    }
    
    @Override
    public boolean supportsWeightedData() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Classifier clone() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public CategoricalResults classify(DataPoint data) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    private  class cluster{
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
