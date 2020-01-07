/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.preprocessing;

import java.util.ArrayList;
import java.util.List;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.PAM;
import jsat.clustering.SeedSelectionMethods;
import jsat.linear.distancemetrics.EuclideanDistance;
import static jsat.preprocessing.PAMBasedUndersampling.getMajorityClassSample;
import static jsat.preprocessing.PAMBasedUndersampling.getMinorityClassSample;
import static jsat.preprocessing.Sampling.sumUp;
import jsat.utils.IntList;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author dell
 */
public class ForOverlapPAMUndersampling {
     public ClassificationDataSet clusteringPAM(ClassificationDataSet dataSet, int numClusters){
        IntList[] classIndex=createDataPartitionBasedOnCategory(dataSet);
        ClassificationDataSet minClass = getMinorityClassSample(dataSet, classIndex);
       // System.out.println("Jumlah sampel kelas +:"+ minClass.getSampleSize());
        ClassificationDataSet  majClass = getMajorityClassSample(dataSet, classIndex);
        
        
        // menjadikan jumlah sampel kelas minoritas sebagai jumlah klaster pada kelas mayoritas
       // int extendedSize = -100; //dibuat antara -100 sd 100
       // int numCluster = minClass.getSampleSize() + extendedSize;
      //  int numCluster = minClass.getSampleSize() ;
        //  int numCluster = 1000;
        PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
       List<List<DataPoint>> clusters1 = pam.cluster(majClass, numClusters);       
        List<List<Integer>> clusters = pam.cluster(dataSet, numClusters, true);
         System.out.println("cluster: "+clusters.size());
        ClassificationDataSet cds = getRandomSelectedDataFromCluster (dataSet,clusters);
        return cds;
        /*
        int [] med= pam.getMedoids();
        
        List<Integer> idx = new ArrayList();
        for(int i=0;i<med.length;i++){
            idx.add(med[i]);
        }
       //identifikasi pusat cluster kelas + atau kelas -
       //jika kelas + maka buang data kelas -
       //jika kelas - maka ambil pusat cluster dan ambil data kelas + 
       
        //   System.out.println("idx med: "+ idx.size());
          
       
           for(int i=0;i<clusters.size();i++){
               int cat= dataSet.getDataPointCategory(idx.get(i));
               int total = clusters.get(i).size();
               int same =0;
               int diff =0;
              // System.out.print(dataSet.getDataPointCategory(idx.get(i))+"; med: "+ idx.get(i)+" elemen cluster: ");
                for(Integer idxi: clusters.get(i)){
                    int catelm = dataSet.getDataPointCategory(idxi);
                    if(catelm == cat){
                        same +=1;
                    }else{
                        diff += 1;
                    }
                    //System.out.print(idxi+": "+ catelm +", " );
                }
               // System.out.println();
               //double dDiff = (double)diff;
               //double dTotal = (double)total;
              // if(dDiff/dTotal<=0.01){
              //  System.out.println(dataSet.getDataPointCategory(idx.get(i))+" diff/total: "+ ((double)diff/(double)total) +" total: "+ total + " same: "+same +" diff: "+diff);
              // }
                
            //   System.out.println("med "+ idx.get(i)+"; cluster elm 1:"+ clusters.get(i).get(0));
           //jika elemen 1 cluster adalah kelas + maka buang kelas negatif dalam cluster tersebut
           //jika elemen 1 cluster adalah kelas - maka ambil elemen pertama dan kelas + didalam cluster tersebut
//           int idxPoint = clusters.get(i).get(0);
           //int category= dataSet.getDataPointCategory(clusters.get(i).get(0));
          // System.out.println("med "+ idx.get(i)+"; category "+ dataSet.getDataPointCategory(idx.get(i)) + "; indeks "+idxPoint+" category "+category);
       }
        
       */
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
   
      //metode sampling data pada cluster
      //jika elemen klaster seragam ==> ambil secara acak
      //jika elemen klaster tidak seragam ==> pilih secara acak kelasnya kemudian pilih secara acak elemennya
      //inputan berupa anggota klaster dan medoidnya
      //luaran berupa data sample
      //inspired by : NCL Lauricala , one side selection, das et al 2013
      public ClassificationDataSet getRandomSelectedDataFromCluster(ClassificationDataSet cds, List<List<Integer>> clusterMember){
      ArrayList<Integer> selectedIndex = new ArrayList();
        //identifikasi keseragaman kelas anggota klaster      
      ArrayList<Integer> category= new ArrayList();      
      for(List<Integer> aI: clusterMember){
          System.out.println("aI size: "+ aI.size());
          boolean isMixed=false;
          if(aI.size()>1){
              int categ = cds.getDataPointCategory(aI.get(0));              
              for(Integer i:aI){
                  int categMem = cds.getDataPointCategory(i);
                  if(categMem!= categ){
                         isMixed = true;                         
                  }
              }
              if(isMixed){
                       category.add(1);
              }else{
                       category.add(0);
              }                            
          }else if(aI.size()==1){
              category.add(0);
          }    
      }
      System.out.println("jumlah category"+ category.size());
      //jika seragam maka ambil secara acak
      //jika tidak seragam maka pilih kelas secara acak lalu pilih member kelas terpilih secara acak
      for(int i=0;i<category.size();i++){
          int categ = category.get(i);
          List<Integer> subcluster = clusterMember.get(i);
          if(categ==0){
          //kondisi seragam
            if(subcluster.size()==1){
                selectedIndex.add(subcluster.get(0));
            }else if(subcluster.size()>1){
                int rdm = RandomUtil.getRandom().nextInt(subcluster.size());
                
                selectedIndex.add(subcluster.get(rdm));
            }
          }else if(categ==1 && subcluster.size()>1){
            //kondisi mix
            int index = getIndexFromMix(cds,subcluster);
            if (index != -1)
                selectedIndex.add(index);
            
          }
      }
      for(int j=0;j<selectedIndex.size();j++){
          System.out.println(selectedIndex.get(j));
      }
      
      ClassificationDataSet newCds = cds.getSubset(selectedIndex);
      return newCds;
      }
      
     private int getIndexFromMix(ClassificationDataSet cds,List<Integer> subCluster ){
         int idx=-1;
         List<Integer> listC1 = new ArrayList();
         List<Integer> listC2 = new ArrayList();
         int firstCateg = cds.getDataPointCategory(subCluster.get(0));
         for(Integer i:subCluster){
            int categMem = cds.getDataPointCategory(i);
            if(categMem!= firstCateg){
                listC1.add(i);
            }else{
                listC2.add(i);
            }
                   
         }
         boolean b = RandomUtil.getRandom().nextBoolean();
         if(b==true){
             if(listC1.size()==1){
                 idx = listC1.get(0);
             }else if(listC1.size()>1){
                 int r = RandomUtil.getRandom().nextInt(listC1.size());
                 idx = listC1.get(r);
             }
             
         }else{
             if(listC2.size()==1){
                 idx = listC2.get(0);
             }else if(listC2.size()>1){
                 int r = RandomUtil.getRandom().nextInt(listC2.size());
                 idx = listC2.get(r);
             }
         }
         return idx;
         
     }
     
     //implementasi metode 2:
     //1. lakukan clustering data
     //2. identifikasi pada setiap klaster apakah terdapat kelas berbeda atau tidak
     //3. ambil data dari kelas seragam dan ambil hanya kelas positif dari kelas campuran
     public ClassificationDataSet moveTheMajorityFromMix(ClassificationDataSet dataSet, int numClusters){
     PAM pam = new PAM(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
       List<List<DataPoint>> clusters1 = pam.cluster(dataSet, numClusters);       
        List<List<Integer>> clusters = pam.cluster(dataSet, numClusters, true);
         System.out.println("cluster: "+clusters.size());
         //identifikasi keseragaman kelas anggota klaster      
      ArrayList<Integer> category= new ArrayList();      
      for(List<Integer> aI: clusters){
          System.out.println("aI size: "+ aI.size());
          boolean isMixed=false;
          if(aI.size()>1){
              int categ = dataSet.getDataPointCategory(aI.get(0));              
              for(Integer i:aI){
                  int categMem = dataSet.getDataPointCategory(i);
                  if(categMem!= categ){
                         isMixed = true;                         
                  }
              }
              if(isMixed){
                       category.add(1);
              }else{
                       category.add(0);
              }                            
          }else if(aI.size()==1){
              category.add(0);
          }    
      }
      //jika kategory seragam maka ambil seluruh data
      //jika kategori campuran maka ambil dari kelas positifnya saja
      //input: categori, clusters dan dataset
      List<Integer> newIdx = new ArrayList();
      for(int idx =0;idx<category.size();idx++){
          int cat = category.get(idx);
          if(cat ==0){
          //ambil semua elemen cluster
          newIdx.addAll(clusters.get(idx));
          }else if (cat ==1){
          //ambil kelas positifnya saja
          List<Integer> list = clusters.get(idx);
          for(Integer i:list){
              if(dataSet.getDataPointCategory(i)== 0){
                  newIdx.add(i);
              }
          }          
          }
      }
      //ambil sampel dari dataSet
      ClassificationDataSet newCds = dataSet.getSubset(newIdx);
      return newCds;
     }
}
