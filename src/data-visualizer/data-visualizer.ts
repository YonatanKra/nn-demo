// @ts-ignore

import template from './data-visualizer.template.html?raw';

import traininingData from '../assets/trainingData.json';
import testData from '../assets/testData.json';

import './data-visualizer.css';
import { addDecisionBoundary, plotData } from './plot/index.js';
import { Neuron } from '../neural-network/neuron.js';

const WEIGHTS = Array(3).fill(0).map(() => Math.random() * 2 - 1);
const BIAS = Math.random()*2 - 1;

export class DataVisualizer extends HTMLElement {
    public static observedAttributes = [];

    neuralNetwork: Neuron;

    get #bigTrainButton() {
        return this.shadowRoot?.querySelector('#bigTrain');
    }

    get #trainButton() {
        return this.shadowRoot?.querySelector('#train');
    }

    get #testButton() {
        return this.shadowRoot?.querySelector('#test');
    }

    get #resetButton() {
        return this.shadowRoot?.querySelector('#resetWeights');
    }

    get #trainWithLearningRateButton() {
        return this.shadowRoot?.querySelector('#trainWithLearningRate');
    }

    get #chartDiv() {
        return this.shadowRoot?.querySelector('#chart');
    }

    constructor() {
        super();
        this.attachShadow({ mode: "open" });
        this.shadowRoot!.innerHTML = template;

        this.#testButton?.addEventListener('click', () => {
            this.#testNetwork();
        });

        this.#trainButton?.addEventListener('click', () => {
            this.neuralNetwork.learningRate = 1;
            this.#trainNetwork(traininingData);
        });

        this.#resetButton?.addEventListener('click', () => {
            this.#resetNetwork();
          });
      
          this.#trainWithLearningRateButton?.addEventListener('click', () => {
            this.neuralNetwork.learningRate = 0.1;
            this.#trainNetwork(traininingData);
          });
    }

    #resetNetwork() {
        this.neuralNetwork.weights = [...WEIGHTS];
        this.neuralNetwork.bias = BIAS;
    }

    #showSuccess(successRate: number) {
        this.shadowRoot!.getElementById('success')!.innerText = `Success: ${successRate}%`;
    }

    #showTrainingError(errors: number, epoch: number) {
        this.shadowRoot!.getElementById('trainingErrors')!.innerText = `Reach ${errors} erros after ${epoch} epochs`;
    }

    #testNetwork() {
        !this.neuralNetwork ? this.neuralNetwork = new Neuron() : '';
        const results = testData.map(input => ({
            prediction: this.neuralNetwork.predict(input.data),
            label: input.label
        }));

        const { projectedData, labels } = this.#plotChart(testData);
        addDecisionBoundary(projectedData, this.neuralNetwork.weights, this.neuralNetwork.bias, this.#chartDiv);
        const success = results.reduce((acc, result) => acc + ((result.prediction === result.label) ? 1 : 0), 0);
        this.#showSuccess(100 * success / testData.length);
    }

    async #trainNetwork(data: typeof traininingData, epochs = 1000) {
        for (let i = 1; i <= epochs; i++) {
            const error = this.neuralNetwork.trainNetwork(data);
            if (error === 0) {
                this.#plotDataWithModel(data);
                this.#showTrainingError(error, i);
                return;
            }
            if (i % 100 === 0) {
                await this.#plotDataWithModel(data);
                this.#showTrainingError(error, i);
            }
        }
    }

    #plotChart(data: any) {
        return plotData(data, this.#chartDiv);
    }

    async #plotDataWithModel(data: typeof traininingData) {
        await new Promise(res => requestAnimationFrame(() => {
            const { projectedData, labels } = this.#plotChart(data);
            addDecisionBoundary(projectedData, this.neuralNetwork.weights, this.neuralNetwork.bias, this.#chartDiv);
            res({ projectedData, labels });
        }));
    }
}
customElements.define('data-visualizer', DataVisualizer);
