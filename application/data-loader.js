// data-loader.js
export class DataLoader {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.featureStats = null;
        this.categoricalMappings = null;
    }

    /**
     * Load CSV data from file
     * @param {string} filePath - Path to the CSV file
     * @returns {Promise<void>}
     */
    async loadCSV(filePath) {
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`Failed to load  ${response.statusText}`);
            }
            const csvText = await response.text();
            this.rawData = this.parseCSV(csvText);
            
            if (!this.rawData || this.rawData.length === 0) {
                throw new Error('No data found in CSV file');
            }
            
            console.log(`Loaded ${this.rawData.length} samples from ${filePath}`);
        } catch (error) {
            console.error('Error loading CSV:', error);
            throw new Error(`CSV loading failed: ${error.message}`);
        }
    }

    /**
     * Parse CSV text into array of objects
     * @param {string} csvText - Raw CSV text
     * @returns {Array<Object>}
     */
    parseCSV(csvText) {
        try {
            const lines = csvText.trim().split('\n');
            if (lines.length < 2) {
                throw new Error('CSV file must contain header and at least one data row');
            }
            
            const headers = lines[0].split(',').map(h => h.trim());
            
            const data = [];
            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',').map(v => v.trim());
                if (values.length !== headers.length) continue;
                
                const row = {};
                headers.forEach((header, index) => {
                    const value = values[index];
                    // Try to parse as number, otherwise keep as string
                    row[header] = isNaN(value) ? value : parseFloat(value);
                });
                data.push(row);
            }
            
            return data;
        } catch (error) {
            throw new Error(`CSV parsing failed: ${error.message}`);
        }
    }

    /**
     * Preprocess  separate features/target, encode categoricals, normalize
     * @returns {Object} Processed train and test data
     */
    preprocessData() {
        if (!this.rawData || this.rawData.length === 0) {
            throw new Error('No data loaded. Call loadCSV first.');
        }

        try {
            // Define feature columns (excluding target and index)
            const numericalFeatures = [
                'top_speed_kmh', 'battery_capacity_kWh', 'torque_nm',
                'acceleration_0_100_s', 'fast_charging_power_kw_dc', 'seats',
                'length_mm', 'width_mm', 'height_mm'
            ];
            
            const categoricalFeatures = [
                'fast_charge_port', 'drivetrain'
            ];
            // Note: battery_type is excluded as it has only one value (no variance)
            
            const targetColumn = 'range_km';

            // Verify required columns exist
            const sampleRow = this.rawData[0];
            const missingColumns = [];
            
            [...numericalFeatures, ...categoricalFeatures, targetColumn].forEach(col => {
                if (!(col in sampleRow)) {
                    missingColumns.push(col);
                }
            });
            
            if (missingColumns.length > 0) {
                throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
            }

            // Extract features and target
            const features = [];
            const targets = [];
            
            this.rawData.forEach(row => {
                if (row[targetColumn] === undefined || row[targetColumn] === null) {
                    return; // Skip rows with missing target
                }
                
                // Check if all required features exist
                const hasAllFeatures = numericalFeatures.every(f => 
                    row[f] !== undefined && row[f] !== null && !isNaN(row[f])
                ) && categoricalFeatures.every(f => 
                    row[f] !== undefined && row[f] !== null
                );
                
                if (hasAllFeatures) {
                    features.push(row);
                    targets.push(row[targetColumn]);
                }
            });

            if (features.length === 0) {
                throw new Error('No valid samples found after filtering missing values');
            }

            console.log(`Valid samples after filtering: ${features.length}`);

            // Build categorical mappings (one-hot encoding preparation)
            this.categoricalMappings = {};
            categoricalFeatures.forEach(feature => {
                const uniqueValues = [...new Set(features.map(f => f[feature]))].sort();
                if (uniqueValues.length === 0) {
                    throw new Error(`No unique values found for categorical feature: ${feature}`);
                }
                this.categoricalMappings[feature] = uniqueValues;
            });

            // Calculate statistics for normalization (mean and std)
            this.featureStats = {};
            numericalFeatures.forEach(feature => {
                const values = features.map(f => f[feature]);
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
                const std = Math.sqrt(variance);
                this.featureStats[feature] = { mean, std: std === 0 ? 1 : std }; // Avoid division by zero
            });

            // Transform features to normalized arrays with one-hot encoding
            const transformedFeatures = features.map(row => 
                this.transformFeatures(row, numericalFeatures, categoricalFeatures)
            );

            // Verify all transformed features have the same shape
            const inputShape = transformedFeatures[0].length;
            const invalidShapes = transformedFeatures.filter(f => f.length !== inputShape);
            if (invalidShapes.length > 0) {
                throw new Error('Feature transformation resulted in inconsistent shapes');
            }

            // Perform 80/20 train/test split
            const splitIndex = Math.floor(features.length * 0.8);
            
            if (splitIndex < 1 || features.length - splitIndex < 1) {
                throw new Error('Dataset too small for train/test split. Need at least 5 samples.');
            }
            
            // Shuffle data before splitting
            const shuffledIndices = this.shuffleIndices(features.length);
            const trainIndices = shuffledIndices.slice(0, splitIndex);
            const testIndices = shuffledIndices.slice(splitIndex);

            const trainFeatures = trainIndices.map(i => transformedFeatures[i]);
            const trainTargets = trainIndices.map(i => targets[i]);
            const testFeatures = testIndices.map(i => transformedFeatures[i]);
            const testTargets = testIndices.map(i => targets[i]);

            this.processedData = {
                train: {
                    features: trainFeatures,
                    targets: trainTargets
                },
                test: {
                    features: testFeatures,
                    targets: testTargets
                },
                inputShape: trainFeatures[0].length,
                featureNames: {
                    numerical: numericalFeatures,
                    categorical: categoricalFeatures
                }
            };

            console.log(`Train samples: ${trainFeatures.length}, Test samples: ${testFeatures.length}`);
            console.log(`Input feature dimension: ${this.processedData.inputShape}`);

            return this.processedData;
            
        } catch (error) {
            console.error('Preprocessing error:', error);
            throw new Error(`Data preprocessing failed: ${error.message}`);
        }
    }

    /**
     * Transform a single row of features
     * @param {Object} row - Raw feature row
     * @param {Array<string>} numericalFeatures - List of numerical feature names
     * @param {Array<string>} categoricalFeatures - List of categorical feature names
     * @returns {Array<number>} Transformed feature vector
     */
    transformFeatures(row, numericalFeatures, categoricalFeatures) {
        const transformed = [];

        // Add normalized numerical features
        numericalFeatures.forEach(feature => {
            const value = row[feature];
            const { mean, std } = this.featureStats[feature];
            const normalized = (value - mean) / std;
            transformed.push(normalized);
        });

        // Add one-hot encoded categorical features
        categoricalFeatures.forEach(feature => {
            const value = row[feature];
            const categories = this.categoricalMappings[feature];
            categories.forEach(category => {
                transformed.push(value === category ? 1 : 0);
            });
        });

        return transformed;
    }

    /**
     * Transform user input features for inference
     * @param {Object} userInput - User input object with feature values
     * @returns {Array<number>} Transformed feature vector
     */
    transformUserInput(userInput) {
        if (!this.featureStats || !this.categoricalMappings) {
            throw new Error('Model not trained. Cannot transform user input.');
        }

        try {
            const numericalFeatures = [
                'top_speed_kmh', 'battery_capacity_kWh', 'torque_nm',
                'acceleration_0_100_s', 'fast_charging_power_kw_dc', 'seats',
                'length_mm', 'width_mm', 'height_mm'
            ];
            
            const categoricalFeatures = ['fast_charge_port', 'drivetrain'];

            return this.transformFeatures(userInput, numericalFeatures, categoricalFeatures);
        } catch (error) {
            throw new Error(`User input transformation failed: ${error.message}`);
        }
    }

    /**
     * Generate shuffled indices for train/test split
     * @param {number} length - Number of samples
     * @returns {Array<number>} Shuffled indices
     */
    shuffleIndices(length) {
        const indices = Array.from({ length }, (_, i) => i);
        
        // Fisher-Yates shuffle
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        return indices;
    }

    /**
     * Get processed data
     * @returns {Object} Processed data object
     */
    getData() {
        if (!this.processedData) {
            throw new Error('Data not preprocessed. Call preprocessData first.');
        }
        return this.processedData;
    }

    /**
     * Get feature statistics for denormalization or inspection
     * @returns {Object} Feature statistics
     */
    getFeatureStats() {
        return this.featureStats;
    }

    /**
     * Get categorical mappings
     * @returns {Object} Categorical mappings
     */
    getCategoricalMappings() {
        return this.categoricalMappings;
    }
}
