// app.js
import { DataLoader } from './data-loader.js';

class EVRangePredictor {
    constructor() {
        this.model = null;
        this.dataLoader = new DataLoader();
        this.isTraining = false;
        this.isModelReady = false;
        this.trainingHistory = { loss: [], valLoss: [], mae: [] };
        this.chart = null;
        
        this.initializeUI();
    }

    /**
     * Initialize UI elements and event listeners
     */
    initializeUI() {
        this.elements = {
            form: document.getElementById('vehicleForm'),
            predictBtn: document.getElementById('predictBtn'),
            trainingStatus: document.getElementById('trainingStatus'),
            progressFill: document.getElementById('progressFill'),
            epochValue: document.getElementById('epochValue'),
            lossValue: document.getElementById('lossValue'),
            valLossValue: document.getElementById('valLossValue'),
            maeValue: document.getElementById('maeValue'),
            predictionResult: document.getElementById('predictionResult'),
            predictionPlaceholder: document.getElementById('predictionPlaceholder'),
            rangeValue: document.getElementById('rangeValue'),
            lossChart: document.getElementById('lossChart')
        };

        // Set up form submission
        this.elements.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.predict();
        });

        // Set up canvas for chart
        this.setupChart();

        // Start application
        this.initialize();
    }

    /**
     * Setup chart canvas
     */
    setupChart() {
        const canvas = this.elements.lossChart;
        const ctx = canvas.getContext('2d');
        const parent = canvas.parentElement;
        
        canvas.width = parent.clientWidth - 30;
        canvas.height = 300;
        
        this.chartContext = ctx;
    }

    /**
     * Initialize the application: load data and train model
     */
    async initialize() {
        try {
            this.updateStatus('Loading training data...', 'loading');
            
            // Load and preprocess data
            await this.dataLoader.loadCSV('data.csv');
            const processedData = this.dataLoader.preprocessData();
            
            this.updateStatus('Building neural network model...', 'loading');
            
            // Build model
            this.buildModel(processedData.inputShape);
            
            this.updateStatus('Training model...', 'loading');
            
            // Train model
            await this.trainModel(processedData);
            
            this.updateStatus('Model ready! Enter vehicle specifications to predict range.', 'success');
            this.isModelReady = true;
            this.elements.predictBtn.disabled = false;
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.updateStatus(`Error: ${error.message}`, 'error');
        }
    }

    /**
     * Build the neural network model
     * @param {number} inputShape - Number of input features
     */
    buildModel(inputShape) {
        try {
            this.model = tf.sequential();

            // Input layer + First hidden layer (32 neurons)
            this.model.add(tf.layers.dense({
                inputShape: [inputShape],
                units: 32,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'hidden1'
            }));

            // Dropout for regularization
            this.model.add(tf.layers.dropout({
                rate: 0.2,
                name: 'dropout1'
            }));

            // Second hidden layer (16 neurons)
            this.model.add(tf.layers.dense({
                units: 16,
                activation: 'relu',
                kernelInitializer: 'heNormal',
                name: 'hidden2'
            }));

            // Dropout for regularization
            this.model.add(tf.layers.dropout({
                rate: 0.15,
                name: 'dropout2'
            }));

            // Output layer (1 neuron for regression)
            this.model.add(tf.layers.dense({
                units: 1,
                activation: 'linear',
                name: 'output'
            }));

            // Compile model
            this.model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'meanSquaredError',
                metrics: ['mae']
            });

            console.log('Model built successfully');
            this.model.summary();
            
        } catch (error) {
            console.error('Error building model:', error);
            throw error;
        }
    }

    /**
     * Train the model
     * @param {Object} data - Processed training data
     */
    async trainModel(data) {
        if (!this.model) {
            throw new Error('Model not built');
        }

        this.isTraining = true;
        this.trainingHistory = { loss: [], valLoss: [], mae: [] };

        try {
            // Convert data to tensors
            const trainX = tf.tensor2d(data.train.features);
            const trainY = tf.tensor2d(data.train.targets, [data.train.targets.length, 1]);
            const testX = tf.tensor2d(data.test.features);
            const testY = tf.tensor2d(data.test.targets, [data.test.targets.length, 1]);

            const totalEpochs = 200;
            const batchSize = 16;

            // Train model with callbacks
            await this.model.fit(trainX, trainY, {
                epochs: totalEpochs,
                batchSize: batchSize,
                validationData: [testX, testY],
                shuffle: true,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // Update training history
                        this.trainingHistory.loss.push(logs.loss);
                        this.trainingHistory.valLoss.push(logs.val_loss);
                        this.trainingHistory.mae.push(logs.mae);

                        // Update UI
                        const progress = ((epoch + 1) / totalEpochs) * 100;
                        this.elements.progressFill.style.width = `${progress}%`;
                        this.elements.progressFill.textContent = `${Math.round(progress)}%`;
                        this.elements.epochValue.textContent = epoch + 1;
                        this.elements.lossValue.textContent = logs.loss.toFixed(2);
                        this.elements.valLossValue.textContent = logs.val_loss.toFixed(2);
                        this.elements.maeValue.textContent = logs.mae.toFixed(2);

                        // Update chart every 5 epochs or on last epoch
                        if ((epoch + 1) % 5 === 0 || epoch === totalEpochs - 1) {
                            this.updateChart();
                        }

                        // Allow UI to update
                        await tf.nextFrame();
                    }
                }
            });

            // Evaluate on test set
            const evalResult = this.model.evaluate(testX, testY);
            const testLoss = await evalResult[0].data();
            const testMAE = await evalResult[1].data();
            
            console.log(`Test Loss: ${testLoss[0].toFixed(2)}, Test MAE: ${testMAE[0].toFixed(2)}`);

            // Clean up tensors
            tf.dispose([trainX, trainY, testX, testY, evalResult]);

            this.isTraining = false;

        } catch (error) {
            console.error('Training error:', error);
            this.isTraining = false;
            throw error;
        }
    }

    /**
     * Update training chart
     */
    updateChart() {
        const ctx = this.chartContext;
        const canvas = this.elements.lossChart;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        if (this.trainingHistory.loss.length === 0) return;
        
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Find max value for scaling
        const allValues = [...this.trainingHistory.loss, ...this.trainingHistory.valLoss];
        const maxValue = Math.max(...allValues);
        const minValue = Math.min(...allValues);
        const range = maxValue - minValue || 1;
        
        // Draw axes
        ctx.strokeStyle = '#dee2e6';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Draw grid lines
        ctx.strokeStyle = '#f1f3f5';
        for (let i = 0; i <= 5; i++) {
            const y = padding + (chartHeight * i) / 5;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }
        
        // Draw training loss
        this.drawLine(ctx, this.trainingHistory.loss, '#667eea', padding, chartWidth, chartHeight, minValue, range);
        
        // Draw validation loss
        this.drawLine(ctx, this.trainingHistory.valLoss, '#764ba2', padding, chartWidth, chartHeight, minValue, range);
        
        // Draw legend
        ctx.font = '12px sans-serif';
        ctx.fillStyle = '#667eea';
        ctx.fillRect(width - padding - 120, padding + 10, 15, 15);
        ctx.fillStyle = '#495057';
        ctx.fillText('Training Loss', width - padding - 100, padding + 22);
        
        ctx.fillStyle = '#764ba2';
        ctx.fillRect(width - padding - 120, padding + 35, 15, 15);
        ctx.fillStyle = '#495057';
        ctx.fillText('Validation Loss', width - padding - 100, padding + 47);
        
        // Draw axis labels
        ctx.fillStyle = '#495057';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Epoch', width / 2, height - 10);
        
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Loss', 0, 0);
        ctx.restore();
        
        // Draw value labels on y-axis
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const y = padding + (chartHeight * i) / 5;
            const value = maxValue - (range * i) / 5;
            ctx.fillText(value.toFixed(0), padding - 10, y + 4);
        }
    }

    /**
     * Draw a line on the chart
     */
    drawLine(ctx, data, color, padding, chartWidth, chartHeight, minValue, range) {
        if (data.length < 2) return;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.forEach((value, index) => {
            const x = padding + (chartWidth * index) / (data.length - 1);
            const normalizedValue = (value - minValue) / range;
            const y = padding + chartHeight - (normalizedValue * chartHeight);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    }

    /**
     * Make prediction based on user input
     */
    async predict() {
        if (!this.isModelReady || this.isTraining) {
            return;
        }

        try {
            // Collect user input
            const userInput = {
                top_speed_kmh: parseFloat(document.getElementById('topSpeed').value),
                battery_capacity_kWh: parseFloat(document.getElementById('batteryCapacity').value),
                torque_nm: parseFloat(document.getElementById('torque').value),
                acceleration_0_100_s: parseFloat(document.getElementById('acceleration').value),
                fast_charging_power_kw_dc: parseFloat(document.getElementById('fastCharging').value),
                fast_charge_port: document.getElementById('chargePort').value,
                seats: parseInt(document.getElementById('seats').value),
                drivetrain: document.getElementById('drivetrain').value,
                length_mm: parseInt(document.getElementById('length').value),
                width_mm: parseInt(document.getElementById('width').value),
                height_mm: parseInt(document.getElementById('height').value)
            };

            // Validate input
            for (const [key, value] of Object.entries(userInput)) {
                if (value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {
                    throw new Error(`Invalid input for ${key}`);
                }
            }

            // Transform user input
            const transformedInput = this.dataLoader.transformUserInput(userInput);

            // Make prediction
            const prediction = tf.tidy(() => {
                const inputTensor = tf.tensor2d([transformedInput]);
                const predictionTensor = this.model.predict(inputTensor);
                return predictionTensor;
            });

            const predictionValue = await prediction.data();
            const rangeKm = Math.round(predictionValue[0]);

            // Clean up
            prediction.dispose();

            // Display result
            this.elements.rangeValue.textContent = rangeKm;
            this.elements.predictionResult.style.display = 'block';
            this.elements.predictionPlaceholder.style.display = 'none';

            console.log(`Predicted range: ${rangeKm} km`);

        } catch (error) {
            console.error('Prediction error:', error);
            alert(`Prediction failed: ${error.message}`);
        }
    }

    /**
     * Update status message
     * @param {string} message - Status message
     * @param {string} type - Status type: 'loading', 'success', 'error'
     */
    updateStatus(message, type) {
        this.elements.trainingStatus.textContent = message;
        this.elements.trainingStatus.className = `status ${type}`;
    }
}

// Initialize application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new EVRangePredictor();
    });
} else {
    new EVRangePredictor();
}
