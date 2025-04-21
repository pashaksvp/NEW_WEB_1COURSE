class DecisionTree {
    constructor() {
        this.tree = null;
    }

    buildTree(data, features, target, depth = 0, maxDepth = 10) {
        if (data.length === 0) return null;
        
        const labels = data.map(row => row[target]);
        if (new Set(labels).size === 1 || depth >= maxDepth) {
            return { type: 'leaf', value: this.mostCommonLabel(labels) };
        }
        
        const bestFeature = this.findBestFeature(data, features, target);
        if (!bestFeature) {
            return { type: 'leaf', value: this.mostCommonLabel(labels) };
        }
        
        const featureValues = [...new Set(data.map(row => row[bestFeature]))];
        const remainingFeatures = features.filter(f => f !== bestFeature);
        
        const children = {};
        for (const value of featureValues) {
            const subset = data.filter(row => row[bestFeature] === value);
            children[value] = this.buildTree(subset, remainingFeatures, target, depth + 1, maxDepth);
        }
        
        return {
            type: 'node',
            feature: bestFeature,
            children: children
        };
    }
    
    findBestFeature(data, features, target) {
        let bestFeature = null;
        let bestGain = -Infinity;
        
        const parentEntropy = this.calculateEntropy(data.map(row => row[target]));
        
        for (const feature of features) {
            const gain = this.calculateInformationGain(data, feature, target, parentEntropy);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feature;
            }
        }
        
        return bestFeature;
    }
    
    calculateInformationGain(data, feature, target, parentEntropy) {
        const featureValues = [...new Set(data.map(row => row[feature]))];
        let childrenEntropy = 0;
        
        for (const value of featureValues) {
            const subset = data.filter(row => row[feature] === value);
            const subsetEntropy = this.calculateEntropy(subset.map(row => row[target]));
            childrenEntropy += (subset.length / data.length) * subsetEntropy;
        }
        
        return parentEntropy - childrenEntropy;
    }
    
    calculateEntropy(labels) {
        const counts = {};
        labels.forEach(label => {
            counts[label] = (counts[label] || 0) + 1;
        });
        
        let entropy = 0;
        const total = labels.length;
        for (const label in counts) {
            const p = counts[label] / total;
            entropy -= p * Math.log2(p);
        }
        
        return isNaN(entropy) ? 0 : entropy;
    }
    
    mostCommonLabel(labels) {
        const counts = {};
        labels.forEach(label => {
            counts[label] = (counts[label] || 0) + 1;
        });
        
        let maxCount = -1;
        let mostCommon = null;
        for (const label in counts) {
            if (counts[label] > maxCount) {
                maxCount = counts[label];
                mostCommon = label;
            }
        }
        
        return mostCommon;
    }
    
    predict(tree, sample) {
        if (tree.type === 'leaf') {
            return { 
                prediction: tree.value, 
                path: [`Leaf: ${tree.value}`] 
            };
        }
        
        const featureValue = sample[tree.feature];
        if (!(featureValue in tree.children)) {
            const possibleValues = Object.keys(tree.children).join(", ");
            return { 
                prediction: null, 
                path: [`Unknown value for feature ${tree.feature}: ${featureValue} (possible values: ${possibleValues})`] 
            };
        }
        
        const childResult = this.predict(tree.children[featureValue], sample);
        return {
            prediction: childResult.prediction,
            path: [`Feature ${tree.feature} = ${featureValue}`, ...childResult.path]
        };
    }
}

let decisionTree = new DecisionTree();
let currentTree = null;
let features = [];
let target = '';

function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return { headers: [], data: [] };
    
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        for (let j = 0; j < headers.length && j < values.length; j++) {
            const val = values[j].trim();
            row[headers[j]] = isNaN(val) ? val : parseFloat(val);
        }
        data.push(row);
    }
    
    return { headers, data };
}

function buildTree() {
    const trainingDataText = document.getElementById('trainingData').value;
    const { headers, data } = parseCSV(trainingDataText);
    
    if (headers.length < 2) {
        alert('CSV must have at least 2 columns (features and label)');
        return;
    }
    
    features = headers.slice(0, -1);
    target = headers[headers.length - 1];
    
    currentTree = decisionTree.buildTree(data, features, target);
    visualizeTree(currentTree);
    
    document.getElementById('inputData').value = features.map(f => "0").join(",");
}

function clearTree() {
    currentTree = null;
    document.getElementById('treeVisualization').innerHTML = '';
    document.getElementById('predictionResult').innerHTML = '';
    document.getElementById('predictionPath').innerHTML = '';
}

function predict() {
    if (!currentTree) {
        alert('Please build a tree first');
        return;
    }
    
    const inputDataText = document.getElementById('inputData').value;
    const inputLines = inputDataText.split('\n').filter(line => line.trim() !== '');
    
    let resultsHtml = '';
    let pathsHtml = '';
    
    for (const line of inputLines) {
        const values = line.split(',').map(v => {
            const trimmed = v.trim();
            return isNaN(trimmed) ? trimmed : parseFloat(trimmed);
        });
        
        if (values.length !== features.length) {
            resultsHtml += `<div>Error: Input has ${values.length} values but expected ${features.length} (${features.join(", ")})</div>`;
            continue;
        }
        
        const sample = {};
        for (let i = 0; i < features.length; i++) {
            sample[features[i]] = values[i];
        }
        
        const { prediction, path } = decisionTree.predict(currentTree, sample);
        const predictionText = prediction !== null ? 
            `<span class="path-node">${prediction}</span>` : 
            `<span style="color:red">Unknown</span>`;
        
        resultsHtml += `<div>Input: ${line} → Prediction: ${predictionText}</div>`;
        pathsHtml += `<div><strong>Path for ${line}:</strong><ul>${
            path.map(step => `<li>${step}</li>`).join('')
        }</ul></div>`;
    }
    
    document.getElementById('predictionResult').innerHTML = resultsHtml;
    document.getElementById('predictionPath').innerHTML = pathsHtml;
}

function visualizeTree(tree) {
    const container = document.getElementById('treeVisualization');
    container.innerHTML = '';
    
    if (!tree) return;
    
    const width = container.clientWidth;
    const height = 400;
    
    const root = convertToD3Hierarchy(tree);
    
    const svg = d3.select(container)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(60, 20)");
    
    const treeLayout = d3.tree().size([width - 120, height - 50]);
    const treeData = treeLayout(d3.hierarchy(root));
    
    svg.selectAll(".link")
        .data(treeData.links())
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", d3.linkVertical()
            .x(d => d.x)
            .y(d => d.y));
    
    const nodes = svg.selectAll(".node")
        .data(treeData.descendants())
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", d => `translate(${d.x},${d.y})`);
    
    nodes.append("circle")
        .attr("r", 10)
        .attr("fill", d => d.data.type === 'leaf' ? "#00FFFF" : "#FF00FF");
    
    nodes.append("text")
        .attr("dy", ".35em")
        .attr("y", d => d.children ? -20 : 20)
        .style("text-anchor", "middle")
        .style("fill", "white")
        .text(d => {
            if (d.data.type === 'leaf') return `${d.data.value}`;
            return `${d.data.feature}`;
        });
    
    svg.selectAll(".edgeLabel")
        .data(treeData.links())
        .enter()
        .append("text")
        .attr("class", "edgeLabel")
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2)
        .attr("dy", "0.35em")
        .attr("text-anchor", "middle")
        .style("fill", "#00FFFF")
        .text(d => d.target.data.edgeValue);
}

function convertToD3Hierarchy(node, parent = null) {
    const d3Node = {
        name: node.type === 'leaf' ? `${node.value}` : `${node.feature}`,
        type: node.type,
        feature: node.feature,
        value: node.value
    };
    
    if (node.type === 'node') {
        d3Node.children = [];
        for (const [value, child] of Object.entries(node.children)) {
            const childNode = convertToD3Hierarchy(child, d3Node);
            childNode.edgeValue = value;
            d3Node.children.push(childNode);
        }
    }
    
    return d3Node;
}