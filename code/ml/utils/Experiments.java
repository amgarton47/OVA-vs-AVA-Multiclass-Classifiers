package ml.utils;

import ml.classifiers.DecisionTreeClassifier;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * A class used to experiment on OVA vs AVA classifier models.
 * 
 * @author aidangarton
 *
 */
public class Experiments {

	public static void main(String[] args) {
		DataSet data = new DataSet("../assign5-starter/data/wines.train", DataSet.TEXTFILE);
		CrossValidationSet cvs = new CrossValidationSet(data, 10);

//		for (int j = 1; j <= 3; j++) {
//
//			double[] accuracies = new double[10];
//
//			for (int i = 0; i < 10; i++) {
//				DataSetSplit split = cvs.getValidationSet(i);
//
//				ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, j);
//				OVAClassifier ova = new OVAClassifier(factory);
//				ova.train(split.getTrain());
//
//				double correct = 0, total = 0;
//				for (Example e : split.getTest().getData()) {
//					total++;
//					if (ova.classify(e) == e.getLabel()) {
//						correct++;
//					}
//				}
//
//				accuracies[i] = correct / total;
//			}
//			
//			System.out.println("Accuracies for OVA w/ DTs of depth " + j);
//
//			for (double i : accuracies) {
//				System.out.println(i);
//			}
//		}

//		for (int j = 1; j <= 3; j++) {
//
//			double[] accuracies = new double[10];
//
//			for (int i = 0; i < 10; i++) {
//				DataSetSplit split = cvs.getValidationSet(i);
//
//				ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, j);
//				AVAClassifier ava = new AVAClassifier(factory);
//				ava.train(split.getTrain());
//
//				double correct = 0, total = 0;
//				for (Example e : split.getTest().getData()) {
//					total++;
//					if (ava.classify(e) == e.getLabel()) {
//						correct++;
//					}
//				}
//
//				accuracies[i] = correct / total;
//			}
//
//			System.out.println("Accuracies for OVA w/ DTs of depth " + j);
//
//			for (double i : accuracies) {
//				System.out.println(i);
//			}
//		}

		double[] accuracies = new double[10];
		double[] totals = new double[10];

		for (int i = 0; i < 10; i++) {
			accuracies[i] = 0;
			totals[i] = 0;
		}

		for (int i = 0; i < 10; i++) {
			DataSetSplit split = cvs.getValidationSet(i);

			System.out.println("Split " + i);

			DecisionTreeClassifier dt = new DecisionTreeClassifier();
			dt.setDepthLimit(7);
			dt.train(split.getTrain());

			double correct = 0, total = 0;
			for (Example e : split.getTest().getData()) {
				total++;
				if (dt.classify(e) == e.getLabel()) {
					correct++;
				}
			}

			accuracies[i] = accuracies[i] + correct;
			totals[i] = totals[i] + total;
		}

		System.out.println("Accuracies for multi-class DT w/ depth 7");

		for (int i = 0; i < 10; i++) {
			System.out.println(accuracies[i] / totals[i]);
		}

	}

}
