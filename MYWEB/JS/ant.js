const canvasElement = document.getElementById('myCanvas');
const context = canvasElement.getContext('2d');
canvasElement.addEventListener('mousedown', onMouseDown);

let antCount = 10;
let iterationCount;
let evaporationRate = 0.5;
let pheromoneMatrix = [];

let alpha = 1;
let beta = 2;
let q = 100;
let initialPheromone = 0.1;

let distances;
let pheromones;

let points = [];

function onMouseDown(event) {
    const xCoordinate = event.clientX - canvasElement.offsetLeft;
    const yCoordinate = event.clientY - canvasElement.offsetTop;

    context.fillStyle = '#FFFFFF';
    context.beginPath();
    context.arc(xCoordinate - 8, yCoordinate - 8, 5, 0, Math.PI * 2);
    context.fill();

    points.push([xCoordinate, yCoordinate]);
}

function calculateDistance(point1, point2) {
    const dx = point2[0] - point1[0];
    const dy = point2[1] - point1[1];
    const distance = Math.sqrt(dx * dx + dy * dy);
    return distance;
}

function createGraph(dots) {
    const dotCount = dots.length;
    const graph = [];

    for (let i = 0; i < dotCount; i++) {
        const distancesFromPoint = [];
        for (let j = 0; j < dotCount; j++) {
            distancesFromPoint.push(calculateDistance(dots[i], dots[j]));
        }
        graph.push(distancesFromPoint);
    }

    return graph;
}

function antColonyOptimization(distanceMatrix, antCount, iterationCount, evaporationRate, alpha, beta, q) {
    pheromoneMatrix = [];
    const initialPheromoneValue = 1 / (distanceMatrix.length * Math.sqrt(distanceMatrix.length));

    for (let i = 0; i < distanceMatrix.length; i++) {
        pheromoneMatrix[i] = new Array(distanceMatrix.length).fill(initialPheromoneValue);
    }

    let bestTour;
    let bestTourLength = Number.MAX_VALUE;

    for (let iteration = 0; iteration < iterationCount; iteration++) {
        const ants = [];
        if (iteration % 5 === 0) {
            animate(pheromoneMatrix, distanceMatrix);
        }

        for (let antIndex = 0; antIndex < antCount; antIndex++) {
            const ant = createAnt(distanceMatrix, pheromoneMatrix, alpha, beta);
            ants.push(ant);
        }

        updatePheromoneLevels(ants, distanceMatrix, pheromoneMatrix, evaporationRate, q);

        ants.forEach((ant) => {
            const tourLength = calculateTourLength(ant.tour, distanceMatrix);
            if (tourLength < bestTourLength) {
                bestTourLength = tourLength;
                bestTour = ant.tour.slice();
            }
        });

        let path = [];
        for (let i = 0; i < bestTour.length; i++) {
            path.push(points[bestTour[i]]);
        }
        if (iteration % 5 === 0) {
            animatePath(context, path);
        }
    }

    return bestTour;
}

function createAnt(distanceMatrix, pheromoneMatrix, alpha, beta) {
    const numCities = distanceMatrix.length;
    const visited = new Set();
    const tour = [];

    const startCity = Math.floor(Math.random() * numCities);
    tour.push(startCity);
    visited.add(startCity);

    while (visited.size < numCities) {
        const currentCity = tour[tour.length - 1];
        const probabilities = [];
        let totalProbability = 0;

        for (let i = 0; i < numCities; i++) {
            if (!visited.has(i)) {
                const pheromoneLevel = pheromoneMatrix[currentCity][i];
                const distance = distanceMatrix[currentCity][i];
                const probability = Math.pow(pheromoneLevel, alpha) * Math.pow(1 / distance, beta);
                probabilities.push({ city: i, probability });
                totalProbability += probability;
            }
        }

        const randomNumber = Math.random() * totalProbability;
        let cumulativeProbability = 0;
        let nextCity;

        for (let i = 0; i < probabilities.length; i++) {
            cumulativeProbability += probabilities[i].probability;
            if (randomNumber <= cumulativeProbability) {
                nextCity = probabilities[i].city;
                break;
            }
        }

        tour.push(nextCity);
        visited.add(nextCity);
    }

    return { tour };
}

function calculateTourLength(tour, distanceMatrix) {
    let tourLength = 0;
    for (let i = 0; i < tour.length - 1; i++) {
        const fromCity = tour[i];
        const toCity = tour[i + 1];
        tourLength += distanceMatrix[fromCity][toCity];
    }

    tourLength += distanceMatrix[tour[tour.length - 1]][tour[0]];
    return tourLength;
}

function updatePheromoneLevels(ants, distanceMatrix, pheromoneMatrix, evaporationRate, q) {
    const deltaMatrix = [];
    for (let i = 0; i < distanceMatrix.length; i++) {
        deltaMatrix[i] = new Array(distanceMatrix.length).fill(0);
    }

    const tourLengths = ants.map((ant) => calculateTourLength(ant.tour, distanceMatrix));

    ants.forEach((ant, antIndex) => {
        const tour = ant.tour;
        const tourLength = tourLengths[antIndex];
        for (let i = 0; i < tour.length - 1; i++) {
            const currentCity = tour[i];
            const nextCity = tour[i + 1];
            deltaMatrix[currentCity][nextCity] += q / tourLength;
            deltaMatrix[nextCity][currentCity] += q / tourLength;
        }
    });

    for (let i = 0; i < pheromoneMatrix.length; i++) {
        for (let j = i + 1; j < pheromoneMatrix.length; j++) {
            pheromoneMatrix[i][j] = (1 - evaporationRate) * pheromoneMatrix[i][j] + deltaMatrix[i][j];
            pheromoneMatrix[j][i] = pheromoneMatrix[i][j];
        }
    }
}

function drawLineThroughPoints(ctx, dots) {
    if (dots.length < 2) {
        return;
    }

    ctx.beginPath();
    ctx.moveTo(dots[0][0] - 8, dots[0][1] - 8);

    for (let i = 1; i < dots.length; i++) {
        ctx.lineTo(dots[i][0] - 8, dots[i][1] - 8);
    }

    ctx.closePath();
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 3;

    ctx.stroke();
}

function drawPoints(ctx, dots) {
    ctx.fillStyle = 'black';
    ctx.moveTo(dots[0][0], dots[0][1]);
    for (let i = 0; i < dots.length; i++) {
        const x = dots[i][0];
        const y = dots[i][1];
        ctx.beginPath();
        ctx.arc(x - 8, y - 8, 5, 0, Math.PI * 2);
        ctx.fill();
    }
}

function drawPath(ctx, path) {
    context.clearRect(0, 0, canvasElement.width, canvasElement.height);
    drawLineThroughPoints(ctx, path);
    drawPoints(ctx, points);
}

function drawPheromoneTrails(pheromoneLevels, distanceMatrix) {
    context.clearRect(0, 0, canvasElement.width, canvasElement.height);
    const maxPheromoneLevel = Math.max(...pheromoneLevels.flat());

    context.lineWidth = 10;
    context.beginPath();

    for (let i = 0; i < distanceMatrix.length; i++) {
        for (let j = i + 1; j < distanceMatrix[i].length; j++) {
            const pheromoneLevel = pheromoneLevels[i][j];
            const opacity = pheromoneLevel / maxPheromoneLevel;

            if (opacity > 0) { 
                context.strokeStyle = `rgba(0, 255, 0, ${opacity})`;
                context.moveTo(points[i][0] - 8, points[i][1] - 8);
                context.lineTo(points[j][0] - 8, points[j][1] - 8);
            }
        }
    }
    
    context.stroke();
    drawPoints(context, points);
}

function animate(pheromoneLevels, distanceMatrix) {
    let animationFrame = 0;
    const maxFrames = 60;

    const animationLoop = setInterval(() => {
        drawPheromoneTrails(pheromoneLevels, distanceMatrix);
        if (++animationFrame >= maxFrames) {
            clearInterval(animationLoop);
        }
    }, 1000 / 30); 
}

function animatePath(context, path) {
    let animationFrame = 0;
    const maxFrames = 60;

    const animationLoop = setInterval(() => {
        drawPath(context, path);
        if (++animationFrame >= maxFrames) {
            clearInterval(animationLoop);
        }
    }, 1000 / 30); 
}

function startButton() {
    iterationCount = document.getElementById("numIterations").value;
    distances = createGraph(points);
    pheromones = new Array(distances.length).fill().map(() => new Array(distances.length).fill(initialPheromone));

    let answer = antColonyOptimization(distances, antCount, iterationCount, evaporationRate, alpha, beta, q);
    let path = [];
    for (let i = 0; i < answer.length; i++) {
        path.push(points[answer[i]]);
    }
    animatePath(context, path);
}

function clearButton() {
    context.clearRect(0, 0, canvasElement.width, canvasElement.height);
    points = [];
}
