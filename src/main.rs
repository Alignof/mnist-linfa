use linfa::composing::MultiClassModel;
use linfa::prelude::*;
use linfa_svm::{error::Result, Svm};
use mnist::{Mnist, MnistBuilder};

fn main() -> Result<()> {
    // use 50k samples from the MNIST dataset
    let data_size: usize = 5000;
    let (rows, cols) = (28, 28);

    println!("start!");

    println!("downloading dataset...");
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(data_size as u32)
        .download_and_extract()
        .finalize();

    println!("preparing dataset");
    let ds = linfa::Dataset::new(
        ndarray::Array::from_shape_vec((data_size, rows * cols), trn_img)
            .unwrap()
            .mapv(|x| (x as f64) / 255.),
        ndarray::Array::from_shape_vec((data_size, 1), trn_lbl).unwrap(),
    );

    let (train, valid) = ds.map_targets(|x| *x as usize).split_with_ratio(0.9);

    println!(
        "Fit SVM classifier with #{} training points",
        train.nsamples()
    );

    let params = Svm::<_, Pr>::params()
        //.pos_neg_weights(5000., 500.)
        .gaussian_kernel(30.0);

    let model = train
        .one_vs_all()?
        .into_iter()
        .map(|(l, x)| (l, params.fit(&x).unwrap()))
        .collect::<MultiClassModel<_, _>>();

    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid)?;

    // Print the confusion matrix
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}

