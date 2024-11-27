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

    learn(input: number[], label: number) {
      const prediction = this.predict(input);
      const error = label - prediction;
      for (let i = 0; i < this.weights.length; i++) {
        this.weights[i] += input[i]*error;
      }
      return error;
    }

    trainNetwork(inputs: {data: number[], label: number}[]) {
        const averageError = inputs.reduce((acc: number, input) => acc + this.learn(input.data, input.label), 0) / (inputs.length * 2);
        return averageError;
    }
}