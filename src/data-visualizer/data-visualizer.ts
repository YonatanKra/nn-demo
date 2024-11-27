// @ts-ignore

import template from './data-visualizer.template.html?raw';

import testData from '../assets/testData.json';

import './data-visualizer.css';
import { addDecisionBoundary, plotData } from './plot/index.js';
import { Neuron } from '../neural-network/neuron.js';

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

  get #chartDiv() {
    return this.shadowRoot?.querySelector('#chart');
  }

  constructor() {
    super();
    this.attachShadow({mode: "open"});
    this.shadowRoot!.innerHTML = template;

    this.#testButton?.addEventListener('click', () => {
      this.#testNetwork();
    });
  }

  #showSuccess(successRate: number) {
    this.shadowRoot!.getElementById('success')!.innerText = `Success: ${successRate}%`;
  }

  #testNetwork() {
    this.neuralNetwork = new Neuron();
    const results = testData.map(input => ({
      prediction: this.neuralNetwork.predict(input.data),
      label: input.label
    }));

    const {projectedData, labels} = this.#plotChart(testData);
    addDecisionBoundary(projectedData, this.neuralNetwork.weights, this.neuralNetwork.bias, this.#chartDiv);
    const success = results.reduce((acc, result) => acc + ((result.prediction === result.label) ? 1 : 0), 0);
    this.#showSuccess(100 * success / testData.length);
  }

  #plotChart(data: any) {
    return plotData(data, this.#chartDiv);
  }


}
customElements.define('data-visualizer', DataVisualizer);
