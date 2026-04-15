const express    = require('express');
const bodyParser = require('body-parser');
const axios      = require('axios');
const path       = require('path');

const app  = express();
const PORT = process.env.PORT || 5000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../views'));
app.use(express.static(path.join(__dirname, '../public')));

// Routes

// Serve the initial landing page containing the sensor data input form
app.get('/', (req, res) => {
    res.render('index', { prediction: null, error: null, BACKEND_URL });
});

// Process incoming sensor data, submit it to the backend AI engine, and render the evaluation results
app.post('/get-prediction', async (req, res) => {
    const formData = req.body;

    const features = {
        Temperature:       parseFloat(formData.Temperature),
        Pressure:          parseFloat(formData.Pressure),
        Speed:             parseFloat(formData.Speed),
        Vibration:         parseFloat(formData.Vibration),
        Humidity:          parseFloat(formData.Humidity),
        Power_Consumption: parseFloat(formData.Power_Consumption),
        Material_Hardness: parseFloat(formData.Material_Hardness)
    };

    console.log('Sending readings to AI Core:', features);

    try {
        const response = await axios.post(`${BACKEND_URL}/predict`, features);
        const result   = response.data;

        // Include the original sensor inputs in the response so our frontend visualizations
        // (such as the radar chart and signal indicators) can display the exact data submitted
        result.inputs = features;
        
        // Ensure the specific machine ID is carried over for reporting
        result.machine_id = req.body.Machine_ID || "Unknown Machine";

        res.render('index', { prediction: result, error: null, BACKEND_URL });

    } catch (err) {
        let errorMessage = 'Could not reach AI Core — is it running?';

        if (err.response) {
            console.error('AI Core returned an error:', err.response.data);
            errorMessage = err.response.data && err.response.data.detail
                ? `AI Error: ${err.response.data.detail}`
                : `AI Server Error (${err.response.status})`;
        } else if (err.request) {
            console.error('No response received from AI Core:', err.message);
        } else {
            console.error('Request setup error:', err.message);
        }

        res.render('index', { prediction: null, error: errorMessage, BACKEND_URL });
    }
});

if (process.env.NODE_ENV !== 'production') {
    app.listen(PORT, () => {
        console.log(`Inspector Pro running at http://localhost:${PORT}`);
    });
}

module.exports = app;
