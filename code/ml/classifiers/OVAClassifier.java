package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import ml.data.DataSet;
import ml.data.Example;

/**
 * A one-vs-all multi-class classifier. Designed to utilize any type of binary
 * classifier as its "sub"-models.
 * 
 * 
 * @author Aidan Garton
 *
 */
public class OVAClassifier implements Classifier {
	private ClassifierFactory factory;
	private ArrayList<Classifier> classifiers;

	/**
	 * Simple 1-param constructor that instantiates factory variable
	 * 
	 * @param factory this generates meta/sub binary models
	 */
	public OVAClassifier(ClassifierFactory factory) {
		this.factory = factory;
	}

	/**
	 * Trains a number of models equal to the number of classes in the train
	 * dataset. Each sub-model corresponds to a one label-vs-rest of the labels
	 * classifier. These are stored in an arraylist for classification use later.
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void train(DataSet data) {
		classifiers = new ArrayList<Classifier>();

		// iterate through the classes present in our data, creating a binary model for
		// each (i.e. current label vs all other labels)
		Iterator<Double> labelIter = data.getLabels().iterator();
		while (labelIter.hasNext()) {
			double labelIndex = labelIter.next();

			// create binary dataset based on current label
			DataSet OVADataSet = new DataSet((HashMap<Integer, String>) data.getFeatureMap().clone());
			ArrayList<Example> dataCopy = (ArrayList<Example>) data.getData().clone();

			for (int j = 0; j < dataCopy.size(); j++) {
				Example e = new Example(dataCopy.get(j));

				e.setLabel(e.getLabel() == labelIndex ? labelIndex : -1);
				OVADataSet.addData(e);
			}

			// train a new classifier for each label
			Classifier c = factory.getClassifier();
			c.train(OVADataSet);
			classifiers.add(c);
			System.out.println("Done training classifier for label: " + labelIndex);

			if (labelIndex == 10) {
				System.out.println(c);
			}
		}
	}

	/**
	 * Classifies a data example by running it through each of the classifiers
	 * created above. It returns the positive label with the highest confidence or
	 * the predicted negative label with the lowest confidence.
	 */
	@Override
	public double classify(Example example) {
		double maxConfidence = -1, minConfidence = 47;
		double labelIndex = -1;
		boolean posPredMade = false;

		// For each classifier, classify the provided example
		for (Classifier c : classifiers) {

			double pred = c.classify(example);
			double confidence = c.confidence(example);
			if (pred != -1) {
				// keep track of highest confidence / label
				if (maxConfidence < confidence) {
					maxConfidence = confidence;
					labelIndex = pred;
					posPredMade = true;
				}

				// keep track of lowest confidence / label
				if (!posPredMade) {
					if (minConfidence > confidence) {
						minConfidence = confidence;
						labelIndex = pred;
					}
				}
			}
		}
		return labelIndex;
	}

	@Override
	public double confidence(Example example) {
		// does it make sense just to return the confidence of whichever model is chosen
		// to classify the Example (i.e., what is found above)?
		return 0;
	}

	public static void main(String[] args) {
		DataSet data = new DataSet("../assign5-starter/data/wines.train", DataSet.TEXTFILE);
		ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 3);
		OVAClassifier ova = new OVAClassifier(factory);
//		ClassifierTimer.timeClassifier(ova, data, 1);
		ova.train(data);
//
//		double correct = 0, total = 0;
//		for (Example e : data.getData()) {
//			total++;
//			if (ova.classify(e) == e.getLabel()) {
//				correct++;
//			}
//		}
//		System.out.println("accuracy: " + correct / total * 100);
//		System.out.println(correct);
//		System.out.println(total);
	}
}
