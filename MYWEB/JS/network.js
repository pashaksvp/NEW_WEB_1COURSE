class NeuralNetwork {
    constructor() {
      this.inputSize = 25;
      this.hiddenSize = 15;
      this.outputSize = 10;
      
      this.weightsInputHidden = this.randomMatrix(this.inputSize, this.hiddenSize, 0.2);
      this.weightsHiddenOutput = this.randomMatrix(this.hiddenSize, this.outputSize, 0.2);
      
      this.biasHidden = new Array(this.hiddenSize).fill(0.1);
      this.biasOutput = new Array(this.outputSize).fill(0.1);
      
      this.learningRate = 0.5;
      
      this.trainingData = this.createEnhancedTrainingData();
      
      this.train(5000);
    }
    
    randomMatrix(rows, cols, scale) {
      return Array.from({length: rows}, () => 
        Array.from({length: cols}, () => (Math.random() * 2 - 1) * scale)
      );
    }
    
    sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
    }
    
    sigmoidDerivative(x) {
      return x * (1 - x);
    }
    
    createEnhancedTrainingData() {
      const basicDigits = [
        [1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1], // 0
        [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0], // 1
        [1,1,1,1,1, 0,0,0,0,1, 1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1], // 2
        [1,1,1,1,1, 0,0,0,0,1, 0,1,1,1,1, 0,0,0,0,1, 1,1,1,1,1], // 3
        [1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,0,0,1, 0,0,0,0,1], // 4
        [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1, 0,0,0,0,1, 1,1,1,1,1], // 5
        [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1, 1,0,0,0,1, 1,1,1,1,1], // 6
        [1,1,1,1,1, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0], // 7
        [1,1,1,1,1, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,1,1,1,1], // 8
        [1,1,1,1,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,0,0,1, 1,1,1,1,1]  // 9
      ];
      
      const variations = [];
      for (let digit = 0; digit < 10; digit++) {
        const basePattern = basicDigits[digit];
        
        variations.push({
          input: [...basePattern],
          output: Array.from({length: 10}, (_, i) => i === digit ? 1 : 0)
        });
        
        for (let v = 0; v < 3; v++) {
          const variation = [...basePattern];
          
          for (let i = 0; i < 5; i++) {
            const idx = Math.floor(Math.random() * 25);
            variation[idx] = variation[idx] > 0 ? 0 : 1;
          }
          
          variations.push({
            input: variation,
            output: Array.from({length: 10}, (_, i) => i === digit ? 1 : 0)
          });
        }
      }
      
      return variations;
    }
    
    forward(input) {
      const normalizedInput = input.map(x => x);
      
      const hidden = new Array(this.hiddenSize).fill(0);
      for (let h = 0; h < this.hiddenSize; h++) {
        let sum = this.biasHidden[h];
        for (let i = 0; i < this.inputSize; i++) {
          sum += normalizedInput[i] * this.weightsInputHidden[i][h];
        }
        hidden[h] = this.sigmoid(sum);
      }
      
      const output = new Array(this.outputSize).fill(0);
      for (let o = 0; o < this.outputSize; o++) {
        let sum = this.biasOutput[o];
        for (let h = 0; h < this.hiddenSize; h++) {
          sum += hidden[h] * this.weightsHiddenOutput[h][o];
        }
        output[o] = this.sigmoid(sum);
      }
      
      return { hidden, output };
    }
    
    train(iterations) {
      for (let epoch = 0; epoch < iterations; epoch++) {
        const data = this.trainingData[epoch % this.trainingData.length];
        const { hidden, output } = this.forward(data.input);
        
        const outputErrors = new Array(this.outputSize);
        for (let o = 0; o < this.outputSize; o++) {
          outputErrors[o] = data.output[o] - output[o];
        }
        
        const outputGradients = new Array(this.outputSize);
        for (let o = 0; o < this.outputSize; o++) {
          outputGradients[o] = outputErrors[o] * this.sigmoidDerivative(output[o]);
        }
        
        const hiddenErrors = new Array(this.hiddenSize);
        for (let h = 0; h < this.hiddenSize; h++) {
          let sum = 0;
          for (let o = 0; o < this.outputSize; o++) {
            sum += outputGradients[o] * this.weightsHiddenOutput[h][o];
          }
          hiddenErrors[h] = sum;
        }
        
        const hiddenGradients = new Array(this.hiddenSize);
        for (let h = 0; h < this.hiddenSize; h++) {
          hiddenGradients[h] = hiddenErrors[h] * this.sigmoidDerivative(hidden[h]);
        }
        
        for (let h = 0; h < this.hiddenSize; h++) {
          for (let o = 0; o < this.outputSize; o++) {
            this.weightsHiddenOutput[h][o] += this.learningRate * outputGradients[o] * hidden[h];
          }
        }
        
        for (let i = 0; i < this.inputSize; i++) {
          for (let h = 0; h < this.hiddenSize; h++) {
            this.weightsInputHidden[i][h] += this.learningRate * hiddenGradients[h] * data.input[i];
          }
        }
        
        for (let o = 0; o < this.outputSize; o++) {
          this.biasOutput[o] += this.learningRate * outputGradients[o];
        }
        
        for (let h = 0; h < this.hiddenSize; h++) {
          this.biasHidden[h] += this.learningRate * hiddenGradients[h];
        }
      }
    }
    
    predict(input) {
      const { output } = this.forward(input);
      return output;
    }
  }
  
  const pixelGrid = document.getElementById('pixelGrid');
  const predictionElement = document.getElementById('prediction');
  const nn = new NeuralNetwork();
  let pixels = Array(25).fill(0);
  
  for (let i = 0; i < 25; i++) {
    const pixel = document.createElement('div');
    pixel.className = 'pixel';
    pixel.dataset.index = i;
    pixel.addEventListener('click', togglePixel);
    pixelGrid.appendChild(pixel);
  }
  
  function togglePixel(e) {
    const index = parseInt(e.target.dataset.index);
    pixels[index] = pixels[index] ? 0 : 1;
    e.target.classList.toggle('active');
    predictDigit(); 
  }
  
  function clearPixels() {
    pixels = Array(25).fill(0);
    document.querySelectorAll('.pixel').forEach(pixel => {
      pixel.classList.remove('active');
    });
    predictionElement.textContent = 'Click pixels to draw a digit (0-9)';
  }
  
  function predictDigit() {
    const outputs = nn.predict(pixels);
    const maxIndex = outputs.indexOf(Math.max(...outputs));
    const confidence = (outputs[maxIndex] * 100).toFixed(1);
    
    let confidenceText = `Predicted: ${maxIndex} (${confidence}%)\n`;
    confidenceText += outputs.map((val, i) => 
      `${i}: ${(val * 100).toFixed(1)}%`
    ).join('  ');
    
    predictionElement.textContent = confidenceText;
  }