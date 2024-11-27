export class Neuron {
    weights = new Array(3).fill(0).map(_ => Math.random() * 2 - 1);
    bias = Math.random() * 2 - 1;
    learningRate = Math.random() * 2 - 1;

    constructor({ inputSize, learningRate }) {
        this.weights = new Array(inputSize).fill(0).map(_ => Math.random() * 2 - 1);
        this.learningRate = learningRate;
    }

    predict(input: number[]) {
        const weightedSum = input.reduce((acc, inputPoint, index) =>
            acc + this.weights[index] * inputPoint, 0) + this.bias;
        return this.activation(weightedSum);
    }

    activation(output: number) {
        return output > 0 ? 1 : -1;
        return output > 0 ? output : 0;
    }

    calculateImpact(output: number) {
        return output > 0 ? 1 : 0;
    }

    learn(input: number[], label: number) {
        const prediction = this.predict(input);
        const error = label - prediction;
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += input[i] * error * this.learningRate;
        }
        this.bias += error;
        return error;
    }

    trainNetwork(inputs: { data: number[], label: number }[]) {
        const averageError = inputs.reduce((acc: number, input) => acc + this.learn(input.data, input.label), 0) / (inputs.length * 2);
        return averageError;
    }
}

export class Layer {
    neurons: Neuron[];

    constructor(inputSize: number, outputSize: number, learningRate: number) {
        this.neurons = Array.from({ length: outputSize }, () => new Neuron({ inputSize, learningRate }));
    }

    predict(inputs: number[]): number[] {
        return this.neurons.map(neuron => neuron.predict(inputs));
    }

    learn(inputs: number[], targets: number[]) {
        this.neurons.forEach((neuron, index) => {
            neuron.learn(inputs, targets[index]);
        });
    }
}

export class NeuralNetwork {
    layers: Layer[];
    learningRate: number = Math.random();

    constructor(layerSizes: number[]) {
        this.layers = [];
        for (let i = 0; i < layerSizes.length - 1; i++) {
            this.layers.push(new Layer(layerSizes[i], layerSizes[i + 1], this.learningRate));
        }
    }

    predict(input: number[]): number[] {
        let activations = input;
        for (const layer of this.layers) {
            activations = layer.predict(activations);
        }
        return activations;
    }

    train(inputs: number[][], outputTargets: number[][], learningRate: number, epochs: number) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < inputs.length; i++) {
                const input = inputs[i];
                const target = outputTargets[i];

                // Forward pass
                let activations = input;
                for (const layer of this.layers) {
                    activations = layer.predict(activations);
                }

                // Backpropagation
                let delta = activations.map((output, i) => output > 0 ? target[i] - output : 0);

                for (let layerIndex = this.layers.length - 1; layerIndex >= 0; layerIndex--) {
                    const layer = this.layers[layerIndex];
                    const previousLayer = layerIndex > 0 ? this.layers[layerIndex - 1] : null;

                    this.#updateWeightsAndBias(layer, learningRate, delta, activations);

                    delta = this.#getErrorOfPreviousLayer(previousLayer, delta, layer, input);
                }
            }
        }
    }

    #updateWeightsAndBias(layer: Layer, learningRate: number, delta: number[], activations: number[]) {
        for (let j = 0; j < layer.neurons.length; j++) {
            const perceptron = layer.neurons[j];
            for (let k = 0; k < perceptron.weights.length; k++) {
                perceptron.weights[k] += learningRate * delta[j] * activations[k];
            }
            perceptron.bias += learningRate * delta[j];
        }
    }

    #getErrorOfPreviousLayer(previousLayer: Layer | null, delta: number[], layer: Layer, input: number[]) {
        if (previousLayer) {
            delta = previousLayer.neurons.map((p: Neuron, i) => {
                return delta.reduce((acc, error, j) => acc + error * layer.neurons[j].weights[i], 0) * p.calculateImpact(previousLayer.predict(input)[i]);
            });
        }
        return delta;
    }
}
