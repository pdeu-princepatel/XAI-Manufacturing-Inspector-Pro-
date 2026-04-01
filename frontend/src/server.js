const express    = require('express');
const bodyParser = require('body-parser');
const axios      = require('axios');
const path       = require('path');

const app  = express();
const PORT = 5000;

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../views'));
app.use(express.static(path.join(__dirname, '../public')));

// Routes

// Landing page → show the input form
app.get('/', (req, res) => {
    res.render('index', { prediction: null, error: null });
});

// Handle form submission → call AI core, render results
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
        const response = await axios.post('http://localhost:8000/predict', features);
        const result   = response.data;

        // Attach the raw sensor inputs so the front-end charts
        // (radar, signal lights) can use the actual submitted values
        result.inputs = features;

        res.render('index', { prediction: result, error: null });

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

        res.render('index', { prediction: null, error: errorMessage });
    }
});

app.listen(PORT, () => {
    console.log(`Inspector Pro running at http://localhost:${PORT}`);
});
