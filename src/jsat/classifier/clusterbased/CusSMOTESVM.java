/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifier.clusterbased;

import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.parameters.Parameterized;

/**
 *
 * @author dell
 */
public class CusSMOTESVM implements Classifier, Parameterized {

    @Override
    public CategoricalResults classify(DataPoint data) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel) {
        
        //data kelas mayoritas dicluster
        
        //data kelas minoritas dioversampling
    }

    @Override
    public boolean supportsWeightedData() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Classifier clone() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
