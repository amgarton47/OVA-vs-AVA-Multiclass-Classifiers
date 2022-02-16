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

	public OVAClassifier(ClassifierFactory factory) {
		this.factory = factory;
	}

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
			System.out.println(c);
		}
	}

	@Override
	public double classify(Example example) {
		double maxConfidence = -1, minConfidence = 47;
		double labelIndex = -1;
		boolean posPredMade = false;

		// For each classifier, classify the provided example. Return the positive label
		// predicted
		// with the highest confidence or the negative label predicted with the lowest
		// confidence
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
		ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 5);
		OVAClassifier ova = new OVAClassifier(factory);
		ova.train(data);

		double correct = 0, total = 0;
		for (Example e : data.getData()) {
			total++;
			if (ova.classify(e) == e.getLabel()) {
				correct++;
			}
		}
		System.out.println("accuracy: " + correct / total * 100);
		System.out.println(correct);
		System.out.println(total);
	}
}
