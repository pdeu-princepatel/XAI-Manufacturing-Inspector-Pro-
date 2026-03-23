const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path');
const app = express();
const PORT = 5000;

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../views'));
app.use(express.static(path.join(__dirname, '../public')));

// Routes

// 1. Landing Page (The Form)
app.get('/', (req, res) => {
    res.render('index', { prediction: null, error: null });
});

// 2. Handle Form Submission
app.post('/get-prediction', async (req, res) => {
    const formData = req.body;

    // Convert string inputs to numbers
    const features = {
        Temperature: parseFloat(formData.Temperature),
        Pressure: parseFloat(formData.Pressure),
        Speed: parseFloat(formData.Speed),
        Vibration: parseFloat(formData.Vibration),
        Humidity: parseFloat(formData.Humidity),
        Power_Consumption: parseFloat(formData.Power_Consumption),
        Material_Hardness: parseFloat(formData.Material_Hardness)
    };

    console.log("Sending data to AI Core:", features);

    try {
        const response = await axios.post('http://localhost:8000/predict', features);
        const result = response.data;

        // Render the page again with results

        res.render('index', { prediction: result, error: null });
    } catch (error) {
        let errorMessage = "Could not connect to AI Core. Is it running?";

        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error("AI Core returned error:", error.response.data);
            if (error.response.data && error.response.data.detail) {
                errorMessage = `AI Error: ${error.response.data.detail}`;
            } else {
                errorMessage = `AI Server Error (${error.response.status})`;
            }
        } else if (error.request) {
            // The request was made but no response was received
            console.error("No response from AI Core:", error.message);
        } else {
            console.error("Error setting up request:", error.message);
        }

        res.render('index', { prediction: null, error: errorMessage });
    }
});

app.listen(PORT, () => {
    console.log(`Web App running on http://localhost:${PORT}`);
});
