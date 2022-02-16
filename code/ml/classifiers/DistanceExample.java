package ml.classifiers;

import ml.data.Example;

public class DistanceExample implements Comparable<DistanceExample> {
	private Example example;
	private double distance;

	public DistanceExample(Example example, double distance) {
		this.example = example;
		this.distance = distance;
	}

	public Example getExample() {
		return example;
	}

	public double getDistance() {
		return distance;
	}

	@Override
	public int compareTo(DistanceExample that) {
		return Double.compare(this.distance, that.distance);
	}
}
