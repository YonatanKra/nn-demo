export class Neuron {
    weights = new Array(3).fill(0).map(_ => Math.random()*2 - 1);

    predict(input: number[]) {
      const weightedSum = input.reduce((acc, inputPoint, index) => 
        acc + this.weights[index] * inputPoint, 0);
      return this.activation(weightedSum);
    }

    activation(output: number) {
      return output > 0 ? 1 : -1;
    }

    learn(input, label) {
      const prediction = this.predict(input);
      const error = label - prediction;
      for (let i = 0; i < this.weights.length; i++) {
        this.weights[i] += input[i]*error;
      }
    }
}