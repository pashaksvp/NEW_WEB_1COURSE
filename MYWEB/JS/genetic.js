const canvas = document.getElementById('idCanvas');
const ctx = canvas.getContext('2d');
const POINT_RADIUS = 5;
let points = [];
let edges = [];
const findShortestPathsButton = document.getElementById('findShortestPathsButton');
const clearButton = document.querySelector('button:last-of-type');
const mutationRate = 0.01;
let isRunning = false;

function initializeCanvas() {
    canvas.addEventListener('click', function(event) {
        if (!isRunning) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            points.push({ x, y });
            drawPoints();
            if (points.length > 1) {
                drawEdges();
            }
        }
    });
}

function drawPoints() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#FFFFFF';
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, POINT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
    });
}

function drawEdges() {
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    edges = [];
    for (let i = 0; i < points.length - 1; i++) {
        for (let j = i + 1; j < points.length; j++) {
            const distanceBetweenPoints = distance(points[i], points[j]);
            if (!edges[i]) edges[i] = [];
            if (!edges[j]) edges[j] = [];
            edges[i][j] = edges[j][i] = distanceBetweenPoints;
            ctx.beginPath();
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[j].x, points[j].y);
            ctx.stroke();
        }
    }
}

function distance(pointA, pointB) {
    const dx = pointA.x - pointB.x;
    const dy = pointA.y - pointB.y;
    return Math.sqrt(dx * dx + dy * dy);
}

async function runGeneticAlgorithm(points) {
    const populationSize = 50;
    const generations = 100;

    let population = [];
    for (let i = 0; i < populationSize; i++) {
        population.push(createIndividual(points));
    }

    let bestSolution = [];
    let bestDistance = Infinity;

    for (let gen = 0; gen < generations; gen++) {
        if (isRunning) await sleep(50);
        population.sort((a, b) => fitness(b) - fitness(a));
        const elite = population.slice(0, 5);

        let newPopulation = elite.slice();
        while (newPopulation.length < populationSize) {
            const parentA = selectParent(population);
            const parentB = selectParent(population);
            const child = crossover(parentA, parentB);
            mutate(child);
            newPopulation.push(child);
        }

        population = newPopulation;
        const currentBest = population.reduce((best, current) => {
            return fitness(current) < fitness(best) ? current : best;
        });

        const currentDistance = calculateTotalDistance(currentBest);

        if (currentDistance < bestDistance) {
            bestDistance = currentDistance;
            bestSolution = currentBest;
        }

        drawPath(currentBest, 'lightblue');
    }

    drawPath(bestSolution, 'red');
    isRunning = false;
}

function createIndividual(points) {
    let individual = [];
    for (let i = 0; i < points.length; i++) {
        individual.push(i);
    }
    for (let i = individual.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [individual[i], individual[j]] = [individual[j], individual[i]];
    }
    return individual;
}

function fitness(path) {
    return 1 / calculateTotalDistance(path);
}

function selectParent(population) {
    const totalFitness = population.reduce((total, individual) => total + fitness(individual), 0);
    let threshold = Math.random() * totalFitness;
    for (let i = 0; i < population.length; i++) {
        threshold -= fitness(population[i]);
        if (threshold <= 0) {
            return population[i];
        }
    }
}

function crossover(parentA, parentB) {
    const start = Math.floor(Math.random() * parentA.length);
    const end = Math.floor(Math.random() * (parentA.length - start)) + start;
    const child = parentA.slice(start, end);
    for (let i = 0; i < parentB.length; i++) {
        const gene = parentB[i];
        if (!child.includes(gene)) {
            child.push(gene);
        }
    }
    return child;
}

function mutate(path) {
    for (let i = 0; i < path.length; i++) {
        if (Math.random() < mutationRate) {
            const j = Math.floor(Math.random() * path.length);
            [path[i], path[j]] = [path[j], path[i]];
        }
    }
}

function calculateTotalDistance(path) {
    let totalDistance = 0;
    for (let i = 0; i < path.length - 1; i++) {
        totalDistance += distance(points[path[i]], points[path[i + 1]]);
    }
    totalDistance += distance(points[path[path.length - 1]], points[path[0]]);
    return totalDistance;
}

function drawPath(path, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[path[0]].x, points[path[0]].y);
    for (let i = 1; i < path.length; i++) {
        ctx.lineTo(points[path[i]].x, points[path[i]].y);
    }
    ctx.closePath();
    ctx.stroke();
}

function findShortestPaths() {
    if (points.length < 2) {
        alert('Please add at least 2 points.');
        return;
    }

    isRunning = true;
    runGeneticAlgorithm(points);
}

findShortestPathsButton.addEventListener('click', findShortestPaths);
clearButton.addEventListener('click', clearCanvas);

function clearCanvas() {
    isRunning = false;
    points = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

initializeCanvas();

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}