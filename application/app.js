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
            buildingSteps: document.getElementById('buildingSteps'),
            trainingProgress: document.getElementById('trainingProgress'),
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
     * Update step status in UI
     * @param {string} stepName - Step identifier
     * @param {string} status - Status: 'pending', 'active', 'complete', 'failed'
     * @param {string} details - Optional details message
     */
    updateStepStatus(stepName, status, details = '') {
        const step = this.elements.buildingSteps.querySelector(`[data-step="${stepName}"]`);
        if (!step) return;
        
        const icon = step.querySelector('.status-icon');
        const text = step.querySelector('.status-text');
        
        // Update icon
        icon.className = `status-icon ${status}`;
        switch (status) {
            case 'pending':
                icon.textContent = '○';
                break;
            case 'active':
                icon.textContent = '⟳';
                break;
            case 'complete':
                icon.textContent = '✓';
                break;
            case 'failed':
                icon.textContent = '✗';
                break;
        }
        
        // Update text if details provided
        if (details) {
            const originalText = text.textContent.split(' - ')[0];
            text.textContent = `${originalText} - ${details}`;
        }
    }

    /**
     * Initialize the application: load data and train model
     */
    async initialize() {
        try {
            // Step 1: Load data
            this.updateStepStatus('load', 'active');
            this.updateStatus('Loading training data from data.csv...', 'loading');
            
            try {
                await this.dataLoader.loadCSV('data.csv');
                this.updateStepStatus('load', 'complete', `${this.dataLoader.rawData.length} samples loaded`);
            } catch (error) {
                this.updateStepStatus('load', 'failed', error.message);
                throw error;
            }
            
            // Step 2: Preprocess data
            this.updateStepStatus('preprocess', 'active');
            this.updateStatus('Preprocessing features and splitting dataset...', 'loading');
            
            let processedData;
            try {
                processedData = this.dataLoader.preprocessData();
                this.updateStepStatus('preprocess', 'complete', 
                    `${processedData.inputShape} features, ${processedData.train.features.length} train samples`);
            } catch (error) {
                this.updateStepStatus('preprocess', 'failed', error.message);
                throw error;
            }
            
            // Step 3: Build model
            this.updateStepStatus('build', 'active');
            this.updateStatus('Building neural network architecture...', 'loading');
            
            try {
                await this.buildModel(processedData.inputShape);
                const paramCount = this.model.countParams();
                this.updateStepStatus('build', 'complete', `${paramCount} parameters`);
            } catch (error) {
                this.updateStepStatus('build', 'failed', error.message);
                throw error;
            }
            
            // Step 4: Train model
            this.updateStepStatus('train', 'active');
            this.updateStatus('Training model on dataset...', 'loading');
            this.elements.trainingProgress.style.display = 'block';
            
            try {
                await this.trainModel(processedData);
                this.updateStepStatus('train', 'complete', 'Training completed successfully');
            } catch (error) {
                this.updateStepStatus('train', 'failed', error.message);
                throw error;
            }
            
            // Success
            this.updateStatus('✓ Model ready! Enter vehicle specifications to predict range.', 'success');
            this.isModelReady = true;
            this.elements.predictBtn.disabled = false;
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.updateStatus(`✗ Initialization failed: ${error.message}`, 'error');
            this.isModelReady = false;
            this.elements.predictBtn.disabled = true;
        }
    }

    /**
     * Build the neural network model
     * @param {number} inputShape - Number of input features
     */
    async buildModel(inputShape) {
        try {
            // Validate input shape
            if (!inputShape || inputShape <= 0) {
                throw new Error('Invalid input shape. Must be a positive integer.');
            }
            
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
            
            // Allow UI to update
            await tf.nextFrame();
            
        } catch (error) {
            console.error('Error building model:', error);
            throw new Error(`Model building failed: ${error.message}`);
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

        let trainX, trainY, testX, testY;

        try {
            // Validate data
            if (!data.train.features || !data.train.targets || 
                data.train.features.length === 0 || data.train.targets.length === 0) {
                throw new Error('Invalid training  empty features or targets');
            }
            
            if (!data.test.features || !data.test.targets || 
                data.test.features.length === 0 || data.test.targets.length === 0) {
                throw new Error('Invalid test  empty features or targets');
            }

            // Convert data to tensors
            try {
                trainX = tf.tensor2d(data.train.features);
                trainY = tf.tensor2d(data.train.targets, [data.train.targets.length, 1]);
                testX = tf.tensor2d(data.test.features);
                testY = tf.tensor2d(data.test.targets, [data.test.targets.length, 1]);
            } catch (error) {
                throw new Error(`Tensor conversion failed: ${error.message}`);
            }

            // Verify tensor shapes
            if (trainX.shape[0] !== trainY.shape[0]) {
                throw new Error(`Training shape mismatch: features=${trainX.shape[0]}, targets=${trainY.shape[0]}`);
            }
            
            if (testX.shape[0] !== testY.shape[0]) {
                throw new Error(`Test shape mismatch: features=${testX.shape[0]}, targets=${testY.shape[0]}`);
            }

            const totalEpochs = 100;
            const batchSize = 16;

            // Train model with callbacks
            await this.model.fit(trainX, trainY, {
                epochs: totalEpochs,
                batchSize: batchSize,
                validationData: [testX, testY],
                shuffle: true,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // Validate logs
                        if (!logs || logs.loss === undefined || logs.val_loss === undefined) {
                            console.warn(`Invalid logs at epoch ${epoch}`);
                            return;
                        }

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
                    },
                    onTrainEnd: async () => {
                        console.log('Training completed');
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
            
            // Clean up tensors if they were created
            if (trainX) tf.dispose(trainX);
            if (trainY) tf.dispose(trainY);
            if (testX) tf.dispose(testX);
            if (testY) tf.dispose(testY);
            
            throw new Error(`Training failed: ${error.message}`);
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
     * @param {string} type - Status type: 'loading', 'success', 'error', 'info'
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
