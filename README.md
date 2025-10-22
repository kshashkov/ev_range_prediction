## Prompts
### Prompt 1:
You are an expert ML-engineer designing a web application for predicting the single-charge range of an electric vehicle based on its technical and user specifications. The application is JavaScript, browser-only, and possible to run on GitHub pages. The computational resources are restricted to the browser's capacity. Nevertheless, it is likely that the data features' effect on the target is not explicitely evident, so you need to use an Artifical Neural Network architecture. Use TensorFlow.js to power your model.
The schema for the data is given as follows.
Target: 'range_km'
Features: 'top_speed_kmh', 'battery_capacity_kWh', 'torque_nm', 'acceleration_0_100_s', 'fast_charging_power_kw_dc', 'fast_charge_port', 'seats', 'drivetrain', 'length_mm', 'width_mm', 'height_mm'
The dataset itself is attached.
Propose an ANN architecture to solve this issue.
### Prompt 2:
Make a JavaScript prototype implementing your solution.

The result should contain 3 files: 
- 'index.xml':
1. Implement the UI
2. Visualise the training progress of the model at launch 
3. Provide user input area where the user can enter the parameters of their car for inference. The parameter set corresponds to non-target features in the dataset
4. Present the range prediction result.
- 'data-loader.js':
1. Load the data from 'data.csv' in workspace directory
2. Separate the target column ('range_km') from the features
3. Perform feature engineering
4. Perform 20/80 train/test split
- 'app.js'
1. Tie UI to data, model, training flow and inference flow.
2. Implement the model
3. Train the model at the launch of the application loading the data with 'data-loader.js' and viaualising the progress
4. Once the model is ready, collect vehicle parameters from UI user input to predict the range of the described vehicle
5. Allow the user to change their inputs and give a new prediction in such a case

Genral requirements:
- All JS files must use tf.js from CDN and ES6 classes/modules; all dependencies must be client-side.
- Code must handle memory disposal, edge/corner cases, and robust error handling for file loading and shape mismatches.
- Use clear English comments.
- Designed for direct deployment on GitHub Pages (no server or Python backend).

Output format:
- Output two code blocks labeled exactly as 'data-loader.js' and 'app.js' along with 'index.html'.
- No explanations, only code inside the code blocks.
