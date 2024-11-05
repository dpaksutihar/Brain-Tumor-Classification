const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const mongoose = require('mongoose');
const fs = require('fs');
const { spawn } = require('child_process');

// Initialize Express App
const app = express();
const PORT = 3000;

// Enable CORS to allow requests from the front end
app.use(cors());

// Middleware to handle JSON and URL-encoded data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/patientDB', { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('Failed to connect to MongoDB', err));

// Create a schema for patients
const patientSchema = new mongoose.Schema({
    name: String,
    age: Number,
    gender: String,
    address: String,
    phone: String,
    remarks: String,
    
    images: [{
        imagePath: String,
        predictedLabel: String,
        outputImagePath: String,
        classProbabilities: Object, // Add field to store probabilities
    }],
}, { timestamps: true });

// Create a schema for treatments
const treatmentSchema = new mongoose.Schema({
    date: Date,
    details: String,
    outcome: String,
    patientId: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' }, 
}, { timestamps: true });

// Create a model for patients
const Patient = mongoose.model('Patient', patientSchema);

// Create a model for treatments
const Treatment = mongoose.model('Treatment', treatmentSchema);

// Set up multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const dir = './uploads';
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir);
        }
        cb(null, dir); // Directory to store uploaded files
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname)); // Unique file name
    }
});
const upload = multer({ storage });

// Serve uploaded files
app.use('/uploads', express.static('uploads'));

// Serve output files
app.use('/output', express.static('output'));

// API endpoint to handle form submission
app.post('/upload', upload.array('images', 10), async (req, res) => {
    const { name, age, gender, address, phone, remarks } = req.body;

    if (!req.files || req.files.length === 0) {
        return res.status(400).send('No files uploaded.');
    }

    console.log('files::', req.files);
    const imageRecords = req.files.map(file => ({
        imagePath: `/uploads/${file.filename}`,
        predictedLabel: 'Pending',
        outputImagePath: '',
        classProbabilities: {}, // Add empty object to store probabilities later
    }));

    console.log("imageRecords:", imageRecords);
    // Create a new patient document
    const patient = new Patient({
        name,
        age,
        gender,
        address,
        phone,
        remarks,
        images: imageRecords,
    });

    try {
        await patient.save();
        console.log('Patient saved successfully:', patient);

        // Process each image with the Python script
        for (let i = 0; i < req.files.length; i++) {
            const file = req.files[i];
            console.log('the path:', file.path);
            const pythonProcess = spawn('python', ['mri.py', file.path]);
            // console.log('python process:', pythonProcess);
            let dataString = '';

            pythonProcess.stdout.on('data', (data) => {
                dataString += data.toString();
            });

            console.log('dataString:', dataString);

            // Wait for the Python process to complete for each image
            await new Promise((resolve, reject) => {
                pythonProcess.on('exit', async (code) => {
                    if (code !== 0) {
                        console.error('Python process exited with code:', code);
                        return reject('Error processing image with Python script.');
                    }

                    try {
                        // Parse the result from the Python script
                        const result = JSON.parse(dataString);
                        const predictedLabel = result.predicted_label || 'Unknown';
                        const outputImagePath = result.segmented_image_path ? result.segmented_image_path.replace('./output/', '') : '';
                        const classProbabilities = result.class_probabilities || {}; // Get class probabilities

                        // Find the class with the highest probability
                        let highestProb = 0;
                        let diagnosis = 'Unknown';
                        for (const className in classProbabilities) {
                            if (classProbabilities[className] > highestProb) {
                                highestProb = classProbabilities[className];
                                diagnosis = className;
                            }
                        }

                        // Update the specific image record in the patient document
                        patient.images[i].predictedLabel = diagnosis;
                        patient.images[i].classProbabilities = classProbabilities; // Store all class probabilities
                        if (outputImagePath) {
                            patient.images[i].outputImagePath = `/output/${path.basename(outputImagePath)}`;
                        }
                        

                        await patient.save(); // Save the updated patient document with each image's details
                        console.log('Patient updated with predictions for image:', file.filename);

                        // Print patient's profile with each image's details after processing
                        console.log(`\nPatient Profile:`);
                        console.log(`Name: ${patient.name}, Age: ${patient.age}, Address: ${patient.address}, Phone: ${patient.phone}`);
                        console.log(`Image: ${file.filename}`);
                        console.log(`  - Predicted Label (Diagnosis): ${diagnosis}`);
                        console.log(`  - Class Probabilities: ${JSON.stringify(classProbabilities)}`);
                        console.log(`  - Segmented Image Path: ${outputImagePath ? `/output/${path.basename(outputImagePath)}` : 'Not Available'}`);

                        resolve();
                    } catch (parseError) {
                        console.error('Error parsing Python script output:', parseError);
                        reject('Error parsing Python script output.');
                    }
                });
            });
        }

        // Return the final response with patient details and output image paths
        res.json({
            message: 'Form submitted successfully!',
            patientId: patient._id,
            data: {
                ...patient._doc,
                images: patient.images, // Include all images with their individual predictions, probabilities, and output paths
            },
        });
    } catch (saveError) {
        console.error('Error saving patient:', saveError);
        res.status(500).send('Error saving patient to the database.');
    }
});

// API endpoints to manage patients
app.get('/api/patients', async (req, res) => {
    try {
        const patients = await Patient.find();
        console.log("Patients List::", patients);
        res.json(patients);
    } catch (err) {
        console.error('Error fetching patients:', err);
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/patients/:id', async (req, res) => {
    const patientId = req.params.id;
    try {
        console.log("the patient id:", patientId);
        const patient = await Patient.findById(patientId);
        if (!patient) {
            return res.status(404).send({ message: 'Patient not found' });
        }
        console.log("the patient data:", patient);
        res.send(patient);
    } catch (error) {
        console.error('Error retrieving patient information:', error);
        res.status(500).send({ message: 'Error retrieving patient information' });
    }
});

app.delete('/api/patients/:id', async (req, res) => {
    const patientId = req.params.id;
    try {
        const patient = await Patient.findByIdAndDelete(patientId);
        if (!patient) {
            return res.status(404).send({ message: 'Patient not found' });
        }
        console.log("patient is deleted!")
        res.send({ message: 'Patient deleted successfully' });
    } catch (error) {
        console.error('Error deleting patient:', error);
        res.status(500).send({ message: 'Error deleting patient' });
    }
});

// APT endpoint to add treatments
app.post('/api/treatment', async (req, res) => {
    const { treatmentDate, treatmentDetails, treatmentOutcome, patientId  } = req.body;

    console.log("treatment data",req.body );
    // Create a new patient document
    const treatment = new Treatment({
        date: treatmentDate,
        details: treatmentDetails,
        outcome: treatmentOutcome,
        patientId
    });

    try {
        await treatment.save();
        console.log('Treatment saved successfully:', treatment);


        // Return the final response with treatment details
        res.json({
            message: 'Form submitted successfully!',
            treatmentId: treatment,
            data: {
                ...treatment._doc,
            },
        });
    } catch (saveError) {
        console.error('Error saving treatment:', saveError);
        res.status(500).send('Error saving treatment to the database.');
    }
})

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});