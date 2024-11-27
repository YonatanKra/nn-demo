// @ts-ignore
import Plotly from 'plotly.js-dist-min';
import numeric from 'numeric';

export function pca(data, dimensions = 2) {
    const means = data[0].data.map((_, i) =>
        data.reduce((sum, row) => sum + row[i], 0) / data.length
    );

    const centered = data.map(row =>
        row.data.map((val, i) => val - means[i])
    );

    const cov = centered[0].map((_, i) =>
        centered[0].map((_, j) =>
            centered.reduce((sum, row) => sum + row[i] * row[j], 0) / (centered.length - 1)
        )
    );

    const { eigenVectors } = numeric.eig(cov);
    const principalComponents = eigenVectors.slice(0, dimensions);

    return centered.map(row =>
        principalComponents.map(pc =>
            pc.reduce((sum, val, i) => sum + val * row[i], 0)
        )
    );
}

// Function to perform random projection
export function randomProjection(data, targetDimensions = 3) {
    const sourceDimensions = data[0].data.length;
    const projectionMatrix = Array(targetDimensions).fill().map(() =>
        Array(sourceDimensions).fill().map(() => Math.random() * 2 - 1)
    );

    return data.map(point => {
        const vector = point.data;
        return projectionMatrix.map(row =>
            row.reduce((sum, val, i) => sum + val * vector[i], 0)
        );
    });
}

export function calculateCentroid(points) {
    const sum = points.reduce((acc, point) => point.map((v, i) => acc[i] + v), Array(points[0].length).fill(0));
    return sum.map(v => v / points.length);
}

export function addAnalyticalBoundary(projectedData, labels, div) {
    // Calculate centroids
    const traces = [];
    const uniqueLabels = [...new Set(labels)];
    const centroids = uniqueLabels.map(label =>
        calculateCentroid(projectedData.filter((_, i) => labels[i] === label))
    );

    // Add separating line/plane
    if (projectedData[0].length === 2) {
        // 2D case: Add a line
        const midpoint = centroids[0].map((v, i) => (v + centroids[1][i]) / 2);
        const perpSlope = -(centroids[1][0] - centroids[0][0]) / (centroids[1][1] - centroids[0][1]);
        const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
        const yValues = xRange.map(x => perpSlope * (x - midpoint[0]) + midpoint[1]);

        traces.push({
            x: xRange,
            y: yValues,
            mode: 'lines',
            name: 'Separator',
            line: { color: 'rgba(0,0,0,0.5)', dash: 'dash' }
        });
    } else if (projectedData[0].length === 3) {
        // 3D case: Add a plane
        const midpoint = centroids[0].map((v, i) => (v + centroids[1][i]) / 2);
        const normal = centroids[1].map((v, i) => v - centroids[0][i]);

        const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
        const yRange = [Math.min(...projectedData.map(p => p[1])), Math.max(...projectedData.map(p => p[1]))];
        const zFunc = (x, y) => midpoint[2] - (normal[0] * (x - midpoint[0]) + normal[1] * (y - midpoint[1])) / normal[2];

        const planePoints = [];
        for (let x = xRange[0]; x <= xRange[1]; x += (xRange[1] - xRange[0]) / 10) {
            for (let y = yRange[0]; y <= yRange[1]; y += (yRange[1] - yRange[0]) / 10) {
                planePoints.push([x, y, zFunc(x, y)]);
            }
        }

        traces.push({
            x: planePoints.map(p => p[0]),
            y: planePoints.map(p => p[1]),
            z: planePoints.map(p => p[2]),
            type: 'mesh3d',
            opacity: 0.5,
            color: 'rgba(0,0,0,0.5)',
            name: 'Separator'
        });
    }

    Plotly.addTraces(div, traces);
}

// Function to add or update the decision boundary
export function addDecisionBoundary(projectedData, weights, bias = 0, div) {
    const dimension = projectedData[0].length;
    let decisionBoundary;

    if (dimension === 2) {
        // 2D case: Add a line
        const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
        const yValues = xRange.map(x => (-weights[0] * x - bias) / weights[1]);

        decisionBoundary = {
            x: xRange,
            y: yValues,
            mode: 'lines',
            name: 'NN Decision Boundary',
            line: { color: 'rgba(0,0,0,0.5)', dash: 'dash' }
        };
    } else if (dimension === 3) {
        // 3D case: Add a plane
        const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
        const yRange = [Math.min(...projectedData.map(p => p[1])), Math.max(...projectedData.map(p => p[1]))];
        const zFunc = (x, y) => (-weights[0] * x - weights[1] * y - bias) / weights[2];

        const planePoints = [];
        for (let x = xRange[0]; x <= xRange[1]; x += (xRange[1] - xRange[0]) / 10) {
            for (let y = yRange[0]; y <= yRange[1]; y += (yRange[1] - yRange[0]) / 10) {
                planePoints.push([x, y, zFunc(x, y)]);
            }
        }

        decisionBoundary = {
            x: planePoints.map(p => p[0]),
            y: planePoints.map(p => p[1]),
            z: planePoints.map(p => p[2]),
            type: 'mesh3d',
            opacity: 0.5,
            color: 'rgba(0,0,0,0.5)',
            name: 'NN Decision Boundary'
        };
    }

    Plotly.addTraces(div, decisionBoundary);
}
// Function to plot the data
export function plotData(data, div) {
    const dimension = data[0].data.length;
    const labels = data.map(row => row.label);

    let projectedData;
    let layout;

    if (dimension <= 3) {
        projectedData = data.map((row: any) => row.data);
        layout = { title: `${dimension}D Data Visualization` };
    } else {
        projectedData = randomProjection(data, 3);
        layout = { title: 'PCA Visualization (Reduced to 3D)' };
    }

    const traces = [...new Set(labels)].map(label => ({
        x: projectedData.filter((_: any, i: any) => labels[i] === label).map((p: any) => p[0]),
        y: projectedData.filter((_: any, i: any) => labels[i] === label).map((p: any) => p[1]),
        z: projectedData[0].length > 2 ? projectedData.filter((_: any, i: any) => labels[i] === label).map((p: any) => p[2]) : undefined,
        mode: 'markers',
        type: projectedData[0].length > 2 ? 'scatter3d' : 'scatter',
        name: `Label ${label}`,
    }));

    Plotly.newPlot(div, traces, layout);

    return { projectedData, labels };
}

export function addLogisticRegressionVisualization(projectedData, weights, bias, div) {
    const dimension = projectedData[0].length;

    // Sigmoid function
    const sigmoid = z => 1 / (1 + Math.exp(-z));

    if (dimension === 2) {
        // 2D case: Create a heatmap and contour
        const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
        const yRange = [Math.min(...projectedData.map(p => p[1])), Math.max(...projectedData.map(p => p[1]))];

        const xValues = [];
        const yValues = [];
        const zValues = [];

        for (let x = xRange[0]; x <= xRange[1]; x += (xRange[1] - xRange[0]) / 100) {
            for (let y = yRange[0]; y <= yRange[1]; y += (yRange[1] - yRange[0]) / 100) {
                xValues.push(x);
                yValues.push(y);
                const z = weights[0] * x + weights[1] * y + bias;
                zValues.push(sigmoid(z));
            }
        }

        const heatmap = {
            x: xValues,
            y: yValues,
            z: zValues,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            opacity: 0.5,
            name: 'Logistic Regression Probabilities'
        };

        const contour = {
            x: xValues,
            y: yValues,
            z: zValues,
            type: 'contour',
            contours: {
                start: 0.5,
                end: 0.5,
                size: 0.1
            },
            line: { width: 2, color: 'red' },
            showscale: false,
            name: 'Decision Boundary (p=0.5)'
        };

        Plotly.addTraces('plotDiv', [heatmap, contour]);

    } else if (dimension === 3) {
        // 3D case: Create a volume and isosurface
        const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
        const yRange = [Math.min(...projectedData.map(p => p[1])), Math.max(...projectedData.map(p => p[1]))];
        const zRange = [Math.min(...projectedData.map(p => p[2])), Math.max(...projectedData.map(p => p[2]))];

        const values = [];
        const steps = 20;

        for (let x = xRange[0]; x <= xRange[1]; x += (xRange[1] - xRange[0]) / steps) {
            const ySlice = [];
            for (let y = yRange[0]; y <= yRange[1]; y += (yRange[1] - yRange[0]) / steps) {
                const zSlice = [];
                for (let z = zRange[0]; z <= zRange[1]; z += (zRange[1] - zRange[0]) / steps) {
                    const logit = weights[0] * x + weights[1] * y + weights[2] * z + bias;
                    zSlice.push(sigmoid(logit));
                }
                ySlice.push(zSlice);
            }
            values.push(ySlice);
        }

        const volume = {
            type: 'volume',
            x: xRange,
            y: yRange,
            z: zRange,
            value: values,
            opacity: 0.1,
            surface: { count: 5 },
            colorscale: 'Viridis',
            name: 'Logistic Regression Probabilities'
        };

        const isosurface = {
            type: 'isosurface',
            x: xRange,
            y: yRange,
            z: zRange,
            value: values,
            isomin: 0.5,
            isomax: 0.5,
            surface: { show: true, count: 1, fill: 0.8 },
            colorscale: [[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']],
            name: 'Decision Boundary (p=0.5)'
        };

        Plotly.addTraces('plotDiv', [volume, isosurface]);
    }

    // Add a color bar to show probability scale
    const layout = {
        coloraxis: { colorbar: { title: 'Probability' } },
    };
    Plotly.relayout(div, layout);
}

function multiLayerPlot(projectedData, weights1, bias1, weights2, bias2, div) {
    // Activation functions
    function sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    function relu(z) {
        return Math.max(0, z);
    }

    // Two-layer model function
    function twoLayerModel(input, weights1, bias1, weights2, bias2) {
        // First layer
        const z1 = input.map((_, i) =>
            input.reduce((sum, val, j) => sum + weights1[i][j] * val, 0) + bias1[i]
        );

        // Activation (ReLU)
        const a1 = z1.map(relu);

        // Output layer
        const z2 = a1.reduce((sum, val, i) => sum + weights2[i] * val, 0) + bias2;

        // Sigmoid activation for output
        return sigmoid(z2);
    }

    // Function to add two-layer model visualization
    function addTwoLayerVisualization(projectedData, weights1, bias1, weights2, bias2, div) {
        const dimension = projectedData[0].length;

        if (dimension === 2) {
            // 2D case
            const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
            const yRange = [Math.min(...projectedData.map(p => p[1])), Math.max(...projectedData.map(p => p[1]))];

            const xValues = [];
            const yValues = [];
            const zValues = [];

            for (let x = xRange[0]; x <= xRange[1]; x += (xRange[1] - xRange[0]) / 100) {
                for (let y = yRange[0]; y <= yRange[1]; y += (yRange[1] - yRange[0]) / 100) {
                    xValues.push(x);
                    yValues.push(y);
                    zValues.push(twoLayerModel([x, y], weights1, bias1, weights2, bias2));
                }
            }

            const heatmap = {
                x: xValues,
                y: yValues,
                z: zValues,
                type: 'heatmap',
                colorscale: 'Viridis',
                showscale: true,
                opacity: 0.5,
                name: 'Two-Layer Model Probabilities'
            };

            const contour = {
                x: xValues,
                y: yValues,
                z: zValues,
                type: 'contour',
                contours: {
                    start: 0.5,
                    end: 0.5,
                    size: 0.1
                },
                line: { width: 2, color: 'red' },
                showscale: false,
                name: 'Decision Boundary (p=0.5)'
            };

            Plotly.addTraces(div, [heatmap, contour]);

        } else if (dimension === 3) {
            // 3D case
            const xRange = [Math.min(...projectedData.map(p => p[0])), Math.max(...projectedData.map(p => p[0]))];
            const yRange = [Math.min(...projectedData.map(p => p[1])), Math.max(...projectedData.map(p => p[1]))];
            const zRange = [Math.min(...projectedData.map(p => p[2])), Math.max(...projectedData.map(p => p[2]))];

            const values = [];
            const steps = 20;

            for (let x = xRange[0]; x <= xRange[1]; x += (xRange[1] - xRange[0]) / steps) {
                const ySlice = [];
                for (let y = yRange[0]; y <= yRange[1]; y += (yRange[1] - yRange[0]) / steps) {
                    const zSlice = [];
                    for (let z = zRange[0]; z <= zRange[1]; z += (zRange[1] - zRange[0]) / steps) {
                        zSlice.push(twoLayerModel([x, y, z], weights1, bias1, weights2, bias2));
                    }
                    ySlice.push(zSlice);
                }
                values.push(ySlice);
            }

            const volume = {
                type: 'volume',
                x: xRange,
                y: yRange,
                z: zRange,
                value: values,
                opacity: 0.1,
                surface: { count: 5 },
                colorscale: 'Viridis',
                name: 'Two-Layer Model Probabilities'
            };

            const isosurface = {
                type: 'isosurface',
                x: xRange,
                y: yRange,
                z: zRange,
                value: values,
                isomin: 0.5,
                isomax: 0.5,
                surface: { show: true, count: 1, fill: 0.8 },
                colorscale: [[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']],
                name: 'Decision Boundary (p=0.5)'
            };

            Plotly.addTraces(div, [volume, isosurface]);
        }

        // Add a color bar to show probability scale
        const layout = {
            coloraxis: { colorbar: { title: 'Probability' } },
        };
        Plotly.relayout('plotDiv', layout);
    }

    

    // Add the two-layer model visualization for 2D
    addTwoLayerVisualization(projectedData, weights1, bias1, weights2, bias2, div);
}