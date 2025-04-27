## Testing the Model

To test the trained model with a sample image:

1.  Ensure you have the necessary libraries installed, Run: `pip install -r requirements.txt`.
2.  Make sure the final trained model (`plant_species_disease_model_final.keras`) and the binarizer files (`binarizers/species_binarizer.pkl`, `binarizers/disease_binarizer.pkl`) are present in the project's root directory and the `binarizers` subdirectory, respectively. These are generated after running `model.py`.
3.  Sample images are provided in the `input/` folder.
4.  Run the prediction script from your terminal in the project's root directory:

    ```bash
    python predict.py
    ```

5.  The script will load the model, preprocess the image located at `input/img.jpeg`, perform the prediction, and print the predicted species and disease along with their confidence scores to the console.
6.  Fix any broken path in the files.
7.  Don't touch the broken_model and the api-test files.
8.  Also don't recommend weak model, because as the name suggests, its weak.