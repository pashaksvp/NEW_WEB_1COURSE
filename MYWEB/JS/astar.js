let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let cellSize = 20; 
let rows = Math.floor(canvas.height / cellSize);
let cols = Math.floor(canvas.width / cellSize);

if (rows % 2 === 0) rows--;
if (cols % 2 === 0) cols--;

let startCell = {row: 1, col: 1};
let endCell = {row: rows - 2, col: cols - 2};
let grid = new Array(rows).fill(0).map(() => new Array(cols).fill(0));

let startSet = false;
let endSet = false;
let wallSet = false;

const colors = {
    empty: '#ffffff',
    wall: '#000',
    start: '#4CAF50',
    end: '#ff0000',
    visited: '#562b19',
    frontier: '#3a0ca3',
    path:'rgb(68, 255, 0)',
    current: '#f72585',
    neighbor: '#7209b7'
};

document.getElementById('setStart').addEventListener('click', function() {
    startSet = true;
    endSet = false;
    wallSet = false;
});

document.getElementById('setEnd').addEventListener('click', function() {
    startSet = false;
    endSet = true;
    wallSet = false;
});

document.getElementById('setWall').addEventListener('click', function() {
    startSet = false;
    endSet = false;
    wallSet = true;
});

document.getElementById('clearCanvas').addEventListener('click', function() {
    grid = new Array(rows).fill(0).map(() => new Array(cols).fill(0));
    startCell = {row: 1, col: 1};
    endCell = {row: rows - 2, col: cols - 2};
    draw();
});

document.getElementById('generateMaze').addEventListener('click', function() {
    generateMaze();
});

document.getElementById('canvas').addEventListener('click', function(e) {
    let x = Math.floor(e.offsetX / cellSize);
    let y = Math.floor(e.offsetY / cellSize);
    
    if (x < 0 || x >= cols || y < 0 || y >= rows) return;
    
    if (startSet) {
        if (grid[y][x] !== 1) {
            startCell = {row: y, col: x};
            draw();
        }
    } else if (endSet) {
        if (grid[y][x] !== 1) {
            endCell = {row: y, col: x};
            draw();
        }
    } else if (wallSet) {
        if (!(y === startCell.row && x === startCell.col) && 
            !(y === endCell.row && x === endCell.col)) {
            grid[y][x] = grid[y][x] === 0 ? 1 : 0;
            draw();
        }
    }
});

document.getElementById('runAlgorithm').addEventListener('click', function() {
    aStar(grid, startCell, endCell).then(path => {
        if (path.length > 0) {
            drawPath(path);
        } else {
            alert("No path found!");
        }
    });
});

function generateMaze() {
    disableUI();
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            grid[i][j] = 1;
        }
    }
    
    let startX = 1 + 2 * Math.floor(Math.random() * ((cols - 1) / 2));
    let startY = 1 + 2 * Math.floor(Math.random() * ((rows - 1) / 2));
    
    let frontiers = [];
    grid[startY][startX] = 0;
    addFrontiers(startX, startY, frontiers);
    
    let interval = setInterval(() => {
        if (frontiers.length === 0) {
            clearInterval(interval);
            
            startCell = findFarthestPoint(startX, startY, true);
            endCell = findFarthestPoint(startCell.col, startCell.row, false);
            
            draw();
            enableUI();
            return;
        }
        
        let index = Math.floor(Math.random() * frontiers.length);
        let cell = frontiers[index];
        frontiers.splice(index, 1);
        
        let neighbors = [];
        if (cell.x > 1 && grid[cell.y][cell.x - 2] === 0) neighbors.push({x: cell.x - 2, y: cell.y});
        if (cell.x < cols - 2 && grid[cell.y][cell.x + 2] === 0) neighbors.push({x: cell.x + 2, y: cell.y});
        if (cell.y > 1 && grid[cell.y - 2][cell.x] === 0) neighbors.push({x: cell.x, y: cell.y - 2});
        if (cell.y < rows - 2 && grid[cell.y + 2][cell.x] === 0) neighbors.push({x: cell.x, y: cell.y + 2});
        
        if (neighbors.length > 0) {
            let neighbor = neighbors[Math.floor(Math.random() * neighbors.length)];
            
            grid[cell.y][cell.x] = 0;
            grid[(neighbor.y + cell.y) / 2][(neighbor.x + cell.x) / 2] = 0;
            
            addFrontiers(cell.x, cell.y, frontiers);
        }
        
        draw();
    }, 10);
}

function addFrontiers(x, y, frontiers) {
    if (x > 1 && grid[y][x - 2] === 1) frontiers.push({x: x - 2, y: y});
    if (x < cols - 2 && grid[y][x + 2] === 1) frontiers.push({x: x + 2, y: y});
    if (y > 1 && grid[y - 2][x] === 1) frontiers.push({x: x, y: y - 2});
    if (y < rows - 2 && grid[y + 2][x] === 1) frontiers.push({x: x, y: y + 2});
}

function findFarthestPoint(x, y, isStart) {
    let visited = new Array(rows).fill(0).map(() => new Array(cols).fill(false));
    let queue = [{x, y, dist: 0}];
    visited[y][x] = true;
    let farthest = {x, y, dist: 0};
    
    while (queue.length > 0) {
        let current = queue.shift();
        
        if (current.dist > farthest.dist && grid[current.y][current.x] === 0) {
            farthest = current;
        }
        
        let directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        for (let [dx, dy] of directions) {
            let nx = current.x + dx;
            let ny = current.y + dy;
            
            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows && 
                !visited[ny][nx] && grid[ny][nx] === 0) {
                visited[ny][nx] = true;
                queue.push({x: nx, y: ny, dist: current.dist + 1});
            }
        }
    }
    
    return isStart ? {row: farthest.y, col: farthest.x} : {row: farthest.y, col: farthest.x};
}

async function aStar(grid, start, end) {
    disableUI();
    
    let openSet = [start];
    let cameFrom = {};
    let gScore = {};
    let fScore = {};
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            gScore[`${i},${j}`] = Infinity;
            fScore[`${i},${j}`] = Infinity;
        }
    }
    
    gScore[`${start.row},${start.col}`] = 0;
    fScore[`${start.row},${start.col}`] = heuristic(start, end);
    
    while (openSet.length > 0) {
        let current = openSet[0];
        for (let node of openSet) {
            if (fScore[`${node.row},${node.col}`] < fScore[`${current.row},${current.col}`]) {
                current = node;
            }
        }
        
        if (!(current.row === start.row && current.col === start.col) && 
            !(current.row === end.row && current.col === end.col)) {
            ctx.fillStyle = colors.current;
            ctx.fillRect(current.col * cellSize, current.row * cellSize, cellSize, cellSize);
            ctx.strokeStyle = '#7c7b7b';
            ctx.strokeRect(current.col * cellSize, current.row * cellSize, cellSize, cellSize);
            await sleep(50);
        }
        
        if (current.row === end.row && current.col === end.col) {
            let path = reconstructPath(cameFrom, current);
            enableUI();
            return path;
        }
        
        openSet = openSet.filter(node => !(node.row === current.row && node.col === current.col));
        
        let neighbors = getNeighbors(grid, current);
        
        for (let neighbor of neighbors) {
            if (!(neighbor.row === start.row && neighbor.col === start.col) && 
                !(neighbor.row === end.row && neighbor.col === end.col)) {
                ctx.fillStyle = colors.neighbor;
                ctx.fillRect(neighbor.col * cellSize, neighbor.row * cellSize, cellSize, cellSize);
                ctx.strokeStyle = '#7c7b7b';
                ctx.strokeRect(neighbor.col * cellSize, neighbor.row * cellSize, cellSize, cellSize);
                await sleep(10);
            }
            
            let tentativeGScore = gScore[`${current.row},${current.col}`] + 1;
            
            if (tentativeGScore < gScore[`${neighbor.row},${neighbor.col}`]) {
                cameFrom[`${neighbor.row},${neighbor.col}`] = current;
                gScore[`${neighbor.row},${neighbor.col}`] = tentativeGScore;
                fScore[`${neighbor.row},${neighbor.col}`] = gScore[`${neighbor.row},${neighbor.col}`] + heuristic(neighbor, end);
                
                if (!openSet.some(node => node.row === neighbor.row && node.col === neighbor.col)) {
                    openSet.push(neighbor);
                    
                    if (!(neighbor.row === start.row && neighbor.col === start.col) && 
                        !(neighbor.row === end.row && neighbor.col === end.col)) {
                        ctx.fillStyle = colors.frontier;
                        ctx.fillRect(neighbor.col * cellSize, neighbor.row * cellSize, cellSize, cellSize);
                        ctx.strokeStyle = '#7c7b7b';
                        ctx.strokeRect(neighbor.col * cellSize, neighbor.row * cellSize, cellSize, cellSize);
                    }
                }
            }
        }
        
        if (!(current.row === start.row && current.col === start.col) && 
            !(current.row === end.row && current.col === end.col)) {
            ctx.fillStyle = colors.visited;
            ctx.fillRect(current.col * cellSize, current.row * cellSize, cellSize, cellSize);
            ctx.strokeStyle = '#7c7b7b';
            ctx.strokeRect(current.col * cellSize, current.row * cellSize, cellSize, cellSize);
        }
    }
    
    enableUI();
    return []; 
}

function reconstructPath(cameFrom, current) {
    let totalPath = [current];
    while (`${current.row},${current.col}` in cameFrom) {
        current = cameFrom[`${current.row},${current.col}`];
        totalPath.unshift(current);
    }
    return totalPath;
}

function drawPath(path) {
    for (let i = 0; i < path.length; i++) {
        let step = path[i];
        if (!(step.row === startCell.row && step.col === startCell.col) && 
            !(step.row === endCell.row && step.col === endCell.col)) {
            ctx.fillStyle = colors.path;
            ctx.fillRect(step.col * cellSize, step.row * cellSize, cellSize, cellSize);
            ctx.strokeStyle = '#7c7b7b';
            ctx.strokeRect(step.col * cellSize, step.row * cellSize, cellSize, cellSize);
        }
    }
}

function heuristic(a, b) {
    return Math.sqrt(Math.pow(a.row - b.row, 2) + Math.pow(a.col - b.col, 2));
}

function getNeighbors(grid, node) {
    let neighbors = [];
    let directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    
    for (let [dr, dc] of directions) {
        let nr = node.row + dr;
        let nc = node.col + dc;
        
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] !== 1) {
            neighbors.push({row: nr, col: nc});
        }
    }
    
    return neighbors;
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            ctx.fillStyle = grid[i][j] === 1 ? colors.wall : colors.empty;
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            ctx.strokeStyle = '#7c7b7b';
            ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
    }
    
    ctx.fillStyle = colors.start;
    ctx.fillRect(startCell.col * cellSize, startCell.row * cellSize, cellSize, cellSize);
    
    ctx.fillStyle = colors.end;
    ctx.fillRect(endCell.col * cellSize, endCell.row * cellSize, cellSize, cellSize);
}

function enableUI() {
    document.getElementById('setStart').disabled = false;
    document.getElementById('setEnd').disabled = false;
    document.getElementById('setWall').disabled = false;
    document.getElementById('runAlgorithm').disabled = false;
    document.getElementById('generateMaze').disabled = false;
    document.getElementById('clearCanvas').disabled = false;
}

function disableUI() {
    document.getElementById('setStart').disabled = true;
    document.getElementById('setEnd').disabled = true;
    document.getElementById('setWall').disabled = true;
    document.getElementById('runAlgorithm').disabled = true;
    document.getElementById('generateMaze').disabled = true;
    document.getElementById('clearCanvas').disabled = true;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

draw();