package jsat.clustering;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.linear.vectorcollection.DefaultVectorCollection;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionUtils;
import jsat.math.OnLineStatistics;
import jsat.utils.concurrent.ParallelUtils;

/**
 * A density-based algorithm for discovering clusters in large spatial databases 
 * with noise (1996) by Martin Ester , Hans-peter Kriegel , Jörg S , Xiaowei Xu
 * 
 * @author Edward Raff
 */
public class DBSCAN extends ClustererBase
{

    private static final long serialVersionUID = 1627963360642560455L;
    
    /**
     * Used by {@link #cluster(DataSet, double, int, VectorCollection,int[]) } 
     * to mark that a data point as not yet been visited. <br>
     * Clusters that have been visited have a value >= 0, that indicates their cluster. Or have the value {@link #NOISE}
     */
    private static final int UNCLASSIFIED = -1;
    /**
     * Used by {@link #expandCluster(int[], DataSet, int, int, double, int, VectorCollection) } 
     * to mark that a data point has been visited, but was considered noise. 
     */
    private static final int NOISE = -2;
    
    /**
     * Factory used to create a vector space of the inputs. 
     * The paired Integer is the vector's index in the original dataset
     */
    private VectorCollection<VecPaired<Vec, Integer> > vc;
    private DistanceMetric dm;
    private double stndDevs = 2.0;

    public void setStndDevs(double stndDevs) {
        this.stndDevs = stndDevs;
    }

    public DBSCAN(DistanceMetric dm, VectorCollection<VecPaired<Vec, Integer>> vc)
    {
        this.dm = dm;
        this.vc = vc;
    }

    public DBSCAN()
    {
        this(new EuclideanDistance());
    }
    
    public DBSCAN(DistanceMetric dm)
    {
        this(dm ,new DefaultVectorCollection<VecPaired<Vec, Integer>>());
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public DBSCAN(DBSCAN toCopy)
    {
        this.vc = toCopy.vc.clone();
        this.dm = toCopy.dm.clone();
        this.stndDevs = toCopy.stndDevs;
    }
    
    
    public List<List<DataPoint>> cluster(DataSet dataSet, int minPts)
    {
        return createClusterListFromAssignmentArray(cluster(dataSet, minPts, (int[])null), dataSet);
    }
    
    public int[] cluster(DataSet dataSet, int minPts, int[] designations)
    {
        return cluster(dataSet, minPts, false, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, boolean parallel, int[] designations)
    {
        return cluster(dataSet, 3, parallel, designations);
    }

    @Override
    public DBSCAN clone()
    {
        return new DBSCAN(this);
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, int minPts, boolean parallel)
    {
        return createClusterListFromAssignmentArray(cluster(dataSet, minPts, parallel, null), dataSet);
    }
    
    public int[] cluster(DataSet dataSet, int minPts, boolean parallel, int[] designations)
    {
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, parallel);
        vc.build(parallel, getVecIndexPairs(dataSet), dm);
        
        
        OnLineStatistics stats = ParallelUtils.run(parallel, dataSet.getSampleSize(), (start, end)->
        {
            OnLineStatistics s = new OnLineStatistics();
            for(int i = start; i < end; i++)
            {
                DataPoint dp = dataSet.getDataPoint(i);
                s.add(vc.search(dp.getNumericalValues(), minPts+1).get(minPts).getPair());
            }
            return s;
        }, (t, u)->t.apply(t, u));
        
        double eps = stats.getMean() + stats.getStandardDeviation()*stndDevs;
        System.out.println("eps: "+ eps);
        return cluster(dataSet, eps, minPts, vc, parallel, designations);
    }

    private List<VecPaired<Vec, Integer>> getVecIndexPairs(DataSet dataSet)
    {
        List<VecPaired<Vec, Integer>> vecs = new ArrayList<>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            vecs.add(new VecPaired<>(dataSet.getDataPoint(i).getNumericalValues(), i));
        return vecs;
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, double eps, int minPts)
    {
        return createClusterListFromAssignmentArray(cluster(dataSet, eps, minPts, (int[]) null), dataSet);
    }
    public List<List<Integer>> cluster(DataSet dataSet, double eps, int minPts,int mode){
        int[] assignments = cluster(dataSet, eps,minPts, (int[]) null);
        
        return createIdxClusterListFromAssignmentArray(assignments,dataSet);
    }
    public int[] cluster(DataSet dataSet, double eps, int minPts, int[] designations)
    {
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
        return cluster(dataSet, eps, minPts, vc, false, designations);
    }
    
    public List<List<DataPoint>> cluster(DataSet dataSet, double eps, int minPts, boolean parallel)
    {
        return createClusterListFromAssignmentArray(cluster(dataSet, eps, minPts, parallel, null), dataSet);
    }
    
    public int[] cluster(DataSet dataSet, double eps, int minPts, boolean parallel, int[] designations)
    {
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, parallel);
        return cluster(dataSet, eps, minPts, vc, parallel, designations);
    }
    
    private int[] cluster(DataSet dataSet, double eps, int minPts, VectorCollection<VecPaired<Vec, Integer>> vc, boolean parallel, int[] pointCats)
    {
        if (pointCats == null)
            pointCats = new int[dataSet.getSampleSize()];
        Arrays.fill(pointCats, UNCLASSIFIED);
        
        vc.build(parallel, getVecIndexPairs(dataSet), dm);
        List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> allNearestNeighbor = VectorCollectionUtils.allEpsNeighbors(vc, dataSet.getDataVectors(), eps, parallel);
        
        int curClusterID = 0;
        for(int i = 0; i < pointCats.length; i++)
        {
            if(pointCats[i] == UNCLASSIFIED)
            {
                //All assignments are done by expandCluster
                if(expandCluster(pointCats, dataSet, i, curClusterID, eps, minPts, allNearestNeighbor))
                    curClusterID++;
            }
        }
        
        return pointCats;
    }
    
    /**
     * 
     * @param pointCats the array to store the cluster assignments in
     * @param dataSet the data set 
     * @param point the current data point we are working on
     * @param clId the current cluster we are working on
     * @param eps the search radius
     * @param minPts the minimum number of points to create a new cluster
     * @param vc the collection to use to search with 
     * @return true if a cluster was expanded, false if the point was marked as noise
     */
    private boolean expandCluster(int[] pointCats, DataSet dataSet, int point, int clId, double eps, int minPts, List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> allNearNeighbors)
    {
        List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> seeds = allNearNeighbors.get(point);
        
        if(seeds.size() < minPts)// no core point
        {
            pointCats[point] = NOISE;
            return false;
        }
        //Else, all points in seeds are density-reachable from Point
        
        List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> results;
        
        pointCats[point] = clId;
        Queue<VecPaired<VecPaired<Vec, Integer>, Double>> workQue = new ArrayDeque<>(seeds);
        while(!workQue.isEmpty())
        {
            VecPaired<VecPaired<Vec, Integer>, Double> currentP = workQue.poll();
            results = allNearNeighbors.get(currentP.getVector().getPair());
            
            if(results.size() >= minPts)
                for(VecPaired<VecPaired<Vec, Integer>, Double> resultP :  results)
                {
                    int resultPIndx = resultP.getVector().getPair();
                    if(pointCats[resultPIndx] < 0)// is UNCLASSIFIED or NOISE
                    {
                        if(pointCats[resultPIndx] == UNCLASSIFIED)
                            workQue.add(resultP);
                        pointCats[resultPIndx] = clId;
                    }
                }
        }
        
        return true;
    }
}
